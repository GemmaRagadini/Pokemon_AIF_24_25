from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from pygad import pygad
from scipy.optimize import linprog
from torch import nn

from vgc.balance.meta import MetaData
from vgc.behaviour import TeamBuildPolicy, BattlePolicy
from vgc.behaviour.BattlePolicies import TypeSelector
from vgc.competition.StandardPkmMoves import STANDARD_MOVE_ROSTER
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, MAX_HIT_POINTS
from vgc.datatypes.Objects import Pkm, PkmTemplate, PkmFullTeam, PkmRoster, PkmTeam, PkmMove
from vgc.datatypes.Types import N_TYPES, N_STATUS, N_ENTRY_HAZARD
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.util.Encoding import one_hot


class RandomTeamBuilder(TeamBuildPolicy):
    """
    Agent that selects teams randomly.
    """

    def __init__(self):
        self.roster = None

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        self.roster = roster

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        n_pkms = len(self.roster)
        members = np.random.choice(n_pkms, 3, False)
        pre_selection: List[PkmTemplate] = [self.roster[i] for i in members]
        team: List[Pkm] = []
        for pt in pre_selection:
            moves: List[int] = np.random.choice(DEFAULT_PKM_N_MOVES, DEFAULT_PKM_N_MOVES, False)
            team.append(pt.gen_pkm(moves))
        return PkmFullTeam(team)


class FixedTeamBuilder(TeamBuildPolicy):
    """
    Agent that always selects the same team.
    """

    def __init__(self):
        self.roster = None

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        self.roster = roster

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        pre_selection: List[PkmTemplate] = self.roster[0:3]
        team: List[Pkm] = []
        for pt in pre_selection:
            team.append(pt.gen_pkm([0, 1, 2, 3]))
        return PkmFullTeam(team)


def run_battles(pkm0, pkm1, agent0, agent1, n_battles):
    wins = [0, 0]
    t0 = PkmTeam([pkm0])
    t1 = PkmTeam([pkm1])
    env = PkmBattleEnv((t0, t1), encode=(agent0.requires_encode(), agent1.requires_encode()))
    for _ in range(n_battles):
        s, _ = env.reset()
        t = False
        while not t:
            a0 = agent0.get_action(s[0])
            a1 = agent1.get_action(s[1])
            s, _, t, _, _ = env.step([a0, a1])
        wins[env.winner] += 1
    return wins


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class IndividualPkmCounter(TeamBuildPolicy):
    """
    Counter the team composition we believe will be selected by an opponent. We disregard synergies in teams as in the
    original algorithms which were tested over pkm GO and look for individual pairwise win rates and coverage.
    Contrary to the source paper, the meta is not the win rate directly but instead the usage rate, which we assume is
    a direct implication of the win rate. We use epistemic reasoning to find the meta counter teams and play in an
    unpredictable fashion.
    Source: https://ieee-cog.org/2021/assets/papers/paper_192.pdf
    """
    matchup_table = None
    n_pkms = -1
    pkms = None

    def __init__(self, agent0: BattlePolicy = TypeSelector(), agent1: BattlePolicy = TypeSelector(), n_battles=10):
        self.agent0 = agent0
        self.agent1 = agent1
        self.n_battles = n_battles
        self.policy = None
        self.ver = -1

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        """
        Instead of storing the roster, we fill a pairwise match up table where each entry has the estimated win rate
        from a row pkm against a column pkm.
        """
        if self.ver < ver:
            self.ver = ver
            IndividualPkmCounter.pkms = []
            for pt in roster:
                IndividualPkmCounter.pkms.append(pt.gen_pkm([0, 1, 2, 3]))
            IndividualPkmCounter.n_pkms = len(roster)
            IndividualPkmCounter.matchup_table = np.zeros((IndividualPkmCounter.n_pkms, IndividualPkmCounter.n_pkms))
            for i, pkm0 in enumerate(IndividualPkmCounter.pkms):
                for j, pkm1 in enumerate(IndividualPkmCounter.pkms[i:]):
                    if j == 0:  # p0 == p1
                        IndividualPkmCounter.matchup_table[i][i] = 0.5
                    else:
                        wins = run_battles(pkm0, pkm1, self.agent0, self.agent1, self.n_battles)
                        IndividualPkmCounter.matchup_table[i][i + j] = wins[0] / self.n_battles
                        IndividualPkmCounter.matchup_table[i + j][i] = wins[1] / self.n_battles
            average_winrate = np.sum(IndividualPkmCounter.matchup_table, axis=1) / IndividualPkmCounter.n_pkms
            # pre compute policy
            self.policy = softmax(average_winrate)

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        members: List[int] = np.random.choice(IndividualPkmCounter.n_pkms, 3, False, p=self.policy)
        return PkmFullTeam([IndividualPkmCounter.pkms[members[0]], IndividualPkmCounter.pkms[members[1]],
                            IndividualPkmCounter.pkms[members[2]]])

def select_next(matchup_table, n_pkms, members, coverage_weight, t=0.5, r=0.5):
    """
    :param coverage_weight: current coverage weight
    :param t: threshold to determine whe have a good coverage in terms of win rate
    :param r: ratio to reduce the weight at next iteration
    """
    average_winrate = np.dot(matchup_table, coverage_weight) / n_pkms
    policy = average_winrate / average_winrate.sum()
    p = np.random.choice(n_pkms, 1, p=policy)[0]
    while p in members:
        p = np.random.choice(n_pkms, 1, p=policy)[0]
    members.append(p)
    if len(members) < 3:
        for i in range(n_pkms):
            if matchup_table[p][i] >= t:
                coverage_weight[i] *= r
            matchup_table[p][i] = 0.
        select_next(matchup_table, n_pkms, members, coverage_weight)


class MaxPkmCoverage(IndividualPkmCounter):
    """
    Similar to IndividualPkmCounter, but we progressively reduce the weight against covered pkm from pkm we already
    selected.
    Source: https://ieeexplore.ieee.org/document/10115492
    """

    def __init__(self):
        super().__init__()

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        coverage_weight = np.array([1.0] * IndividualPkmCounter.n_pkms)
        members: List[int] = []
        matchup_table = deepcopy(IndividualPkmCounter.matchup_table)
        select_next(matchup_table, IndividualPkmCounter.n_pkms, members, coverage_weight)
        return PkmFullTeam([IndividualPkmCounter.pkms[members[0]],
                            IndividualPkmCounter.pkms[members[1]],
                            IndividualPkmCounter.pkms[members[2]]])


def get_policy(matchup_table, n_units):
    c = np.array([1] + [0] * n_units)
    A_ub = np.column_stack((np.array([[1] * n_units]).transpose(), matchup_table))
    b_ub = np.array([0] * n_units)
    A_eq = np.array([[0] + [1] * n_units])
    b_eq = np.array([1])
    bounds = [(None, None)] + [(0, None)] * n_units
    return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs', bounds=bounds).x[1:]


def minimax_select_next(matchup_table, n_pkms, n_team_members: int = 3):
    policy = get_policy(matchup_table, n_pkms)
    policy /= policy.sum()  # normalize
    return np.random.choice(n_pkms, n_team_members, False, p=policy).tolist()


class MinimaxBuilder(IndividualPkmCounter):
    """
    We maximize coverage assuming our opponent is also, and therefore we try to approximate an equilibrium, as our
    and the opponent's optimal policy is the same.
    Source: https://ieeexplore.ieee.org/document/10115492
    """

    def __init__(self):
        super().__init__()
        self.policy = None

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        super().set_roster(roster, ver)
        policy = get_policy(IndividualPkmCounter.matchup_table, IndividualPkmCounter.n_pkms)
        policy /= policy.sum()
        self.policy = policy

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        members: List[int] = np.random.choice(IndividualPkmCounter.n_pkms, 3, False, p=self.policy)
        return PkmFullTeam([self.pkms[members[0]], self.pkms[members[1]], self.pkms[members[2]]])


def get_meta_teams(matchup_table: Optional[np.ndarray], pkms: Optional[List[Pkm]], meta: MetaData, threshold: int = 10,
                   n_candidates: int = 10) -> Tuple[List[PkmFullTeam], List[float]]:
    n_teams = meta.get_n_teams()
    meta_teams: List[PkmFullTeam] = []
    usage_rate: List[float] = []
    if n_teams >= threshold:
        # UsageRate
        for i in range(n_teams):
            u = 0.0
            team = meta.get_team(i)
            for pkm in team.pkm_list:
                u += meta.get_global_pkm_usage(pkm.pkm_id)
            usage_rate.append(u)
        policy = softmax(np.array(usage_rate))
        indices = np.random.choice(n_teams, n_candidates, p=policy)
        for i in indices:
            meta_teams.append(meta.get_team(i))
        usage_rate = [usage_rate[i] for i in indices]
    else:
        # MinimaxBuilder
        n_pkms = len(pkms)
        for i in range(n_candidates):
            members: List[int] = minimax_select_next(matchup_table, n_pkms)
            team = PkmFullTeam([pkms[members[0]], pkms[members[1]], pkms[members[2]]])
            meta_teams.append(team)
        usage_rate = [1.0] * n_candidates
    return meta_teams, usage_rate


class GAConfigs:

    def __init__(self):
        self.num_generations = 100
        self.num_parents_mating = 2
        self.sol_per_pop = 8
        self.num_genes = 3
        self.init_range_low = 0
        self.init_range_high = 99
        self.parent_selection_type = "tournament"
        self.keep_parents = 1
        self.crossover_type = "two_points"
        self.mutation_type = "random"
        self.mutation_num_genes = 1


def qmapping(qnet, team, pkms, k=5) -> List[int]:
    qmap = qnet(torch.tensor([team]))
    qpkm = np.zeros(len(pkms))
    offset = len(STANDARD_MOVE_ROSTER)
    for i in range(len(pkms)):
        for move in pkms[i].moves:
            qpkm[i] += qmap[0][STANDARD_MOVE_ROSTER.index(move)]
        qpkm[i] += qmap[0][int(pkms[i].type) + offset]
    return torch.topk(torch.from_numpy(qpkm), k)[1]


def encode_move(e, move: PkmMove):
    e += [(move.power / MAX_HIT_POINTS) * move.acc if move.fixed_damage == 0.0 else move.fixed_damage / MAX_HIT_POINTS,
          move.priority,
          move.prob,
          move.target,
          move.recover / MAX_HIT_POINTS,
          move.stat.value,
          move.stage / 2]
    e += one_hot(move.type, N_TYPES)
    e += one_hot(move.status, N_STATUS)
    e += [move.weather != move.weather.CLEAR]
    e += one_hot(move.hazard, N_ENTRY_HAZARD)


def encode_pkm(e, pkm: Pkm):
    e += [pkm.max_hp / MAX_HIT_POINTS]
    e += one_hot(pkm.type, N_TYPES)
    for move in pkm.moves:
        encode_move(e, move)


def encode_full_team(team: PkmFullTeam):
    e = []
    for pkm in team.pkm_list:
        encode_pkm(e, pkm)
    return e


FULL_TEAM_ENCODE_LEN = len(encode_full_team(PkmFullTeam()))

def get_counter(opponent_teams, usage, pkms: List[Pkm], mlp, conf: GAConfigs):
    encoded_teams = []
    for team in opponent_teams:
        encoded_teams.append(encode_full_team(team))

    def fitness_counter_team(sol, solution_idx):
        my_team = PkmFullTeam([pkms[sol[0]], pkms[sol[1]], pkms[sol[2]]])
        em = encode_full_team(my_team)
        fitness = torch.empty((len(encoded_teams)), dtype=torch.float)
        with torch.no_grad():
            for i, e_opp_team in enumerate(encoded_teams):
                fitness[i] = torch.sigmoid(mlp(torch.Tensor([em + e_opp_team])))[0][0].item()
        return torch.dot(fitness, usage).item()

    conf.init_range_high = len(pkms) - 1
    ga_instance = pygad.GA(num_generations=conf.num_generations,
                           num_parents_mating=conf.num_parents_mating,
                           fitness_func=fitness_counter_team,
                           sol_per_pop=conf.sol_per_pop,
                           num_genes=conf.num_genes,
                           init_range_low=conf.init_range_low,
                           init_range_high=conf.init_range_high,
                           parent_selection_type=conf.parent_selection_type,
                           keep_parents=conf.keep_parents,
                           crossover_type=conf.crossover_type,
                           mutation_type=conf.mutation_type,
                           mutation_num_genes=conf.mutation_num_genes,
                           allow_duplicate_genes=False,
                           mutation_by_replacement=True,
                           random_mutation_min_val=conf.init_range_low,
                           random_mutation_max_val=conf.init_range_high,
                           gene_type=int)
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    return solution, solution_fitness


class PredictorMLP(nn.Module):
    """Multilayer Perceptron."""

    def __init__(self, team_size=3):
        super().__init__()
        encode_len = (FULL_TEAM_ENCODE_LEN // 6) * team_size
        self.layers = nn.Sequential(
            # layer 1
            nn.Linear(encode_len * 2, encode_len * 4),
            nn.Dropout(p=0.1),
            # nn.BatchNorm1d(encode_len * 4),
            nn.Tanh(),
            # layer 2
            nn.Linear(encode_len * 4, encode_len * 2),
            nn.Dropout(p=0.1),
            # nn.BatchNorm1d(encode_len * 2),
            nn.Tanh(),
            # layer 3
            nn.Linear(encode_len * 2, encode_len),
            nn.Dropout(p=0.1),
            # nn.BatchNorm1d(encode_len),
            nn.Tanh(),
            # layer 4
            nn.Linear(encode_len, encode_len // 2),
            nn.Dropout(p=0.1),
            # nn.BatchNorm1d(encode_len),
            nn.Tanh(),
            # layer out
            nn.Linear(encode_len // 2, 2),
            # nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)


class MetaCounter(IndividualPkmCounter):
    """
    We find an optimal team against the current meta using an evolutionary strategy.
    Source: https://ieeexplore.ieee.org/document/10115492
    """

    def __init__(self, conf: GAConfigs, n_teams_threshold: int = 10, n_candidates: int = 10):
        super().__init__()
        self.mlp = None
        self.conf = conf
        self.n_teams_threshold = n_teams_threshold
        self.n_candidates = n_candidates

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        super().set_roster(roster, ver)
        self.mlp = PredictorMLP()
        self.mlp.load_state_dict(torch.load('model/team_pred'))
        self.mlp.eval()

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        meta_teams, usage_rate = get_meta_teams(self.matchup_table, self.pkms, meta, self.n_teams_threshold,
                                                self.n_candidates)
        ids, _ = get_counter(meta_teams, torch.tensor(usage_rate), self.pkms, self.mlp, self.conf)
        return PkmFullTeam([self.pkms[ids[0]], self.pkms[ids[1]], self.pkms[ids[2]]])


def get_winrate_sim(team0: PkmFullTeam, team1: PkmFullTeam):
    # TODO
    pass


def get_winrate_pred(mlp: PredictorMLP, team0: np.ndarray, team1: np.ndarray):
    return torch.sigmoid(mlp(torch.Tensor([team0 + team1])))[0][0].item()


class MaxTeamCoverage(IndividualPkmCounter):
    """
    We find optimal teams against individual opponent teams and maximize coverage assuming our opponent is also, and
    therefore we try to approximate an equilibrium, as our and the opponent's optimal policy should be similar.
    Source: https://ieeexplore.ieee.org/document/10115492
    """

    def __init__(self, conf: GAConfigs, n_teams_threshold: int = 10, n_candidates: int = 10, must_encode: bool = True):
        super().__init__()
        self.mlp = None
        self.qnet = None
        self.conf = conf
        self.n_teams_threshold = n_teams_threshold
        self.n_candidates = n_candidates
        self.must_encode = must_encode

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        super().set_roster(roster, ver)
        if self.must_encode:
            self.mlp = PredictorMLP()
            self.mlp.load_state_dict(torch.load('model/team_pred'))
            self.mlp.eval()

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        meta_teams, _ = get_meta_teams(self.matchup_table, self.pkms, meta, self.n_teams_threshold, self.n_candidates)
        counter_teams = []
        for team in meta_teams:
            usage = torch.ones(1)
            ids, _ = get_counter([team], usage, self.pkms, self.mlp, self.conf)
            counter_teams.append(PkmFullTeam([self.pkms[ids[0]], self.pkms[ids[1]], self.pkms[ids[2]]]))
        encoded_teams: List[Union[PkmFullTeam, np.ndarray]] = []
        all_teams = meta_teams + counter_teams
        if self.must_encode:
            for team in all_teams:
                encoded_teams.append(np.array(encode_full_team(team)))
        else:
            encoded_teams = all_teams
        n_teams = len(encoded_teams)
        team_matchup_table = np.zeros((n_teams, n_teams))
        for i, team0 in enumerate(encoded_teams):
            for j, team1 in enumerate(encoded_teams[i:]):
                if j == 0:  # t0 == t1
                    team_matchup_table[i][i] = 0.5
                else:
                    if self.must_encode:
                        winrate = get_winrate_pred(self.mlp, team0, team1)
                    else:
                        winrate = get_winrate_sim(team0, team1)
                    team_matchup_table[i][i + j] = winrate
                    team_matchup_table[i + j][i] = 1.0 - winrate
        policy = get_policy(team_matchup_table, n_teams)
        policy /= sum(policy)
        p: int = np.random.choice(n_teams, 1, p=policy)
        return all_teams[p]

class TerminalTeamBuilder(TeamBuildPolicy):
    """
    Terminal interface.
    """

    def __init__(self):
        self.roster = None

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        self.roster = roster

    def get_action(self, s: MetaData) -> PkmFullTeam:
        print('~ Roster ~')
        for i, pt in enumerate(self.roster):
            print(i, '->', pt)
        print(f'Select action in the format p p p with p in [0-{len(self.roster) - 1}]')
        while True:
            valid = True
            try:
                t = input('Select Action: ')
                t = t.split()
                if len(t) != 3:
                    print('Invalid action. Select again.')
                    continue
                for m in t:
                    if not m.isdigit() and 0 < int(m) < len(self.roster):
                        print('Invalid action. Select again.')
                        valid = False
                        break
                if valid:
                    break
            except:
                print('Invalid action. Select again.')
        print()
        pre_selection: List[PkmTemplate] = [self.roster[int(t[0])], self.roster[int(t[1])], self.roster[int(t[2])]]
        team: List[Pkm] = []
        for pt in pre_selection:
            team.append(pt.gen_pkm([0, 1, 2, 3]))
        return PkmFullTeam(team)

# class GUITeamBuild TODO