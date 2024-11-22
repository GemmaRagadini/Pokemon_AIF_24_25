import os

from abc import ABC
from typing import List

import numpy as np

from vgc.balance.meta import MetaData
from vgc.behaviour import BattlePolicy, TeamBuildPolicy
from vgc.datatypes.Objects import Pkm, PkmTemplate, GameState, PkmTeam, PkmFullTeam, PkmRoster, PkmMove


import torch
import torch.nn as nn
import torch.nn.functional as F


class PooledLinear(nn.Module):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.fc0 = nn.Linear(units, units, bias=False)
        self.fc1 = nn.Linear(units, units, bias=False)
        self.fc2 = nn.Linear(units, units, bias=False)
        self.bn = nn.BatchNorm1d(units)

    def forward(self, x):
        h0 = self.fc0(x)
        h1 = self.fc1(torch.max(x, 1)[0]).unsqueeze(1)
        h2 = self.fc2(torch.mean(x, 1)).unsqueeze(1)
        h = h0 + h1 + h2
        h = self.bn(h.flatten(0, 1)).unflatten(0, (-1, x.size(1)))
        return F.relu_(h)


class HigmonNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = 4
        units = 64

        # encoding
        self.move_embedding = nn.Embedding(128, 16, padding_idx=0)
        self.type_embedding = nn.Embedding(64, 8, padding_idx=0)
        self.status_embedding = nn.Embedding(16, 6, padding_idx=0)
        self.weather_embedding = nn.Embedding(8, 4, padding_idx=0)

        # battle
        self.fc = nn.Linear(104, units)
        self.blocks = nn.ModuleList([PooledLinear(units) for _ in range(layers)])
        self.p = nn.Linear(units, 4 + 1)  # attack x 4, switch
        self.v1 = nn.Linear(units * 2, units)
        self.v2 = nn.Linear(units, 1, bias=False)

        # team buiding
        self.fc_t = nn.Linear(75, units)
        self.blocks_t = nn.ModuleList([PooledLinear(units) for _ in range(layers)])
        self.p1h_t = nn.Linear(units * 2, units)
        self.p1a_t = nn.Linear(75, units, bias=False)
        self.p2_t = nn.Linear(units, units // 2)
        self.p3_t = nn.Linear(units // 2, 1)
        self.v1_t = nn.Linear(units * 2, units)
        self.v2_t = nn.Linear(units, 1, bias=False)


    def forward(self, obs, **kwargs):
        if not kwargs.get('team_build'):
            move_embs = self.move_embedding(obs['fid'][:, :4]).permute(0, 2, 1, 3).flatten(2, 3)
            type_embs = self.type_embedding(obs['fid'][:, 4])
            status_embs = self.status_embedding(obs['fid'][:, 5])
            weather_emb = self.weather_embedding(obs['sid']).flatten(1, 2)
            e = torch.cat([obs['s'].unsqueeze(1).repeat(1, 6, 1), weather_emb.unsqueeze(1).repeat(1, 6, 1), obs['f'], move_embs, type_embs, status_embs], 2)

            h = F.relu_(self.fc(e))
            for block in self.blocks:
                h = block(h)

            h_p = h[:, :3]
            p = self.p(h_p).flatten(1, 2)

            h_v = torch.cat([torch.max(h, 1)[0], torch.mean(h, 1)], 1)
            h_v = F.relu_(self.v1(h_v))
            v = torch.tanh_(self.v2(h_v))

        else:
            move_embs = self.move_embedding(obs['fid'][:, :4]).permute(0, 2, 1, 3).flatten(2, 3)
            type_embs = self.type_embedding(obs['fid'][:, 4])
            e = torch.cat([obs['f'], move_embs, type_embs], 2)

            h = F.relu_(self.fc_t(e))
            for block in self.blocks_t:
                h = block(h)
            h = torch.cat([torch.max(h, 1)[0], torch.mean(h, 1)], 1)

            h_p = F.relu_(self.p1h_t(h).unsqueeze(1) + self.p1a_t(e))
            h_p = F.relu_(self.p2_t(h_p))
            p = self.p3_t(h_p).squeeze(2)

            h_v = F.relu_(self.v1_t(h))
            v = torch.tanh_(self.v2_t(h_v))

        return {'policy': p, 'value': v}


def legal_actions(gs):
    actions = [0, 1, 2, 3]
    for i, pkm in enumerate(gs.teams[0].party):
        if pkm.hp > 0:
            actions.append((i + 1) * 5 + 4)
    return actions


def mstanh(x, a):
    return 2 / (1 + np.exp(-x / a)) - 1


def make_battle_observation(gs):
    obs = {}
    pkms = [gs.teams[0].active] + gs.teams[0].party + [gs.teams[1].active] + gs.teams[1].party

    obs['fid'] = np.stack([
        [pkm.moves[0].move_id + 1 for pkm in pkms],
        [pkm.moves[1].move_id + 1 for pkm in pkms],
        [pkm.moves[2].move_id + 1 for pkm in pkms],
        [pkm.moves[3].move_id + 1 for pkm in pkms],
        [pkm.type + 1 for pkm in pkms],
        [pkm.status + 1 for pkm in pkms],
    ]).astype(np.int64)

    obs['f'] = np.stack([
        [1] * 3 + [0] * 3,
        [1, 0, 0, 1, 0, 0],
        [mstanh(pkm.hp, 1) for pkm in pkms],
        [mstanh(pkm.hp, 10) for pkm in pkms],
        [mstanh(pkm.hp, 100) for pkm in pkms],
        [mstanh(pkm.max_hp, 1) for pkm in pkms],
        [mstanh(pkm.max_hp, 10) for pkm in pkms],
        [mstanh(pkm.max_hp, 100) for pkm in pkms],
        [pkm.hp / pkm.max_hp for pkm in pkms],
        [mstanh(pkm.n_turns_asleep, 1) for pkm in pkms],
        [mstanh(pkm.n_turns_asleep, 10) for pkm in pkms],
    ], -1).astype(np.float32)

    deads = len([pkm for pkm in pkms[:3] if pkm.hp <= 0]), len([pkm for pkm in pkms[3:] if pkm.hp <= 0])

    obs['sid'] = np.stack([
        gs.weather.condition + 1,
    ]).astype(np.int64)

    obs['s'] = np.stack([
        gs.switched[0],
        gs.switched[1],
        mstanh(gs.turn, 1),
        mstanh(gs.turn, 10),
        mstanh(gs.turn, 100),
        mstanh(gs.weather.n_turns_no_clear, 1),
        mstanh(gs.weather.n_turns_no_clear, 10),
        deads[0] >= 1,
        deads[0] >= 2,
        deads[1] >= 1,
        deads[1] >= 2,
    ]).astype(np.float32)

    return obs


def make_teambuild_observation(roster, selected_ids):
    obs = {}

    obs['fid'] = np.stack([
        [pkm.moves[0].move_id + 1 for pkm in roster],
        [pkm.moves[1].move_id + 1 for pkm in roster],
        [pkm.moves[2].move_id + 1 for pkm in roster],
        [pkm.moves[3].move_id + 1 for pkm in roster],
        [pkm.type + 1 for pkm in roster],
    ]).astype(np.int64)

    obs['f'] = np.stack([
        [mstanh(pkm.max_hp, 1) for pkm in roster],
        [mstanh(pkm.max_hp, 10) for pkm in roster],
        [mstanh(pkm.max_hp, 100) for pkm in roster],
    ], -1).astype(np.float32)

    obs['s'] = np.stack([
        [i in selected_ids for i, _ in enumerate(roster)],
    ]).astype(np.float32)

    return obs


net = HigmonNet()
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "net.pth")
net.load_state_dict(torch.load(path))
net.eval()


class HigmonPlayer(BattlePolicy):
    def __init__(self):
        super().__init__()

    def get_action(self, g: GameState) -> int:
        obs = make_battle_observation(g)
        inputs = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        with torch.inference_mode():
            outputs = net(inputs)
        policy = outputs['policy'].numpy()[0]
        value = outputs['value'].numpy()[0]
        print('policy =', policy)
        print('value =', value)
        actions = legal_actions(g)
        print(actions)
        print([(a, policy[a]) for a in actions])
        action = sorted([(a, policy[a]) for a in actions], key=lambda x: -x[1])[0][0]
        print('selected', action)
        if action % 5 == 4:
            return action // 5 - 1
        else:
            return action % 5


class HigmonTeamBuilder(TeamBuildPolicy):
    def __init__(self):
        self.roster: PkmRoster = None

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        self.roster = roster

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        selected_ids = []
        team: List[Pkm] = []
        for _ in range(3):
            obs = make_teambuild_observation(self.roster, selected_ids)
            inputs = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.inference_mode():
                outputs = net(inputs, team_build=True)
            policy = outputs['policy'].numpy()[0]
            value = outputs['value'].numpy()[0]
            print('policy =', policy)
            print('value =', value)
            action = sorted([(a, policy[a]) for a, _ in enumerate(self.roster) if a not in selected_ids], key=lambda x: -x[1])[0][0]
            print('selected', action)
            selected_ids.append(action)
            team.append(self.roster[action].gen_pkm([0, 1, 2, 3]))

        return PkmFullTeam(team)


battle_policy = HigmonPlayer()
team_build_policy = HigmonTeamBuilder()


class HigmonCompetitor(ABC):
    @property
    def battle_policy(self) -> BattlePolicy:
        return battle_policy

    @property
    def team_build_policy(self) -> TeamBuildPolicy:
        return team_build_policy

    @property
    def name(self) -> str:
        return "Higmon"
