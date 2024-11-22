import numpy as np
from copy import deepcopy
from typing import List
from vgc.balance.meta import MetaData
from vgc.behaviour import TeamBuildPolicy, BattlePolicy
from vgc.behaviour.BattlePolicies import TypeSelector
from vgc.datatypes.Objects import PkmTeam, PkmFullTeam, PkmRoster
from vgc.engine.PkmBattleEnv import PkmBattleEnv

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
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class IndividualPkmCounter(TeamBuildPolicy):
    matchup_table = None
    n_pkms = -1
    pkms = None

    def __init__(self, agent0: BattlePolicy = TypeSelector(), agent1: BattlePolicy = TypeSelector(), n_battles=10, sample_rate=0.3):
        self.agent0 = agent0
        self.agent1 = agent1
        self.n_battles = n_battles
        self.sample_rate = sample_rate
        self.policy = None
        self.ver = -1

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        if self.ver < ver:
            self.ver = ver
            IndividualPkmCounter.pkms = [pt.gen_pkm([0, 1, 2, 3]) for pt in roster]
            IndividualPkmCounter.n_pkms = len(roster)
            IndividualPkmCounter.matchup_table = np.zeros((IndividualPkmCounter.n_pkms, IndividualPkmCounter.n_pkms))
            
            # 상위 티어 포켓몬 선정
            top_tier_indices = self.select_top_tier_indices()
            
            for i, pkm0 in enumerate(IndividualPkmCounter.pkms):
                if i in top_tier_indices:
                    for j in range(i, IndividualPkmCounter.n_pkms):
                        if i == j:
                            IndividualPkmCounter.matchup_table[i][j] = 0.5
                        else:
                            pkm1 = IndividualPkmCounter.pkms[j]
                            wins = run_battles(pkm0, pkm1, self.agent0, self.agent1, self.n_battles)
                            IndividualPkmCounter.matchup_table[i][j] = wins[0] / self.n_battles
                            IndividualPkmCounter.matchup_table[j][i] = wins[1] / self.n_battles
                else:
                    # 중요하지 않은 포켓몬에 대해서는 샘플링
                    sampled_opponents = np.random.choice(IndividualPkmCounter.n_pkms, int(IndividualPkmCounter.n_pkms * self.sample_rate), replace=False)
                    for j in sampled_opponents:
                        if i == j:
                            IndividualPkmCounter.matchup_table[i][j] = 0.5
                        else:
                            pkm1 = IndividualPkmCounter.pkms[j]
                            wins = run_battles(pkm0, pkm1, self.agent0, self.agent1, self.n_battles)
                            IndividualPkmCounter.matchup_table[i][j] = wins[0] / self.n_battles
                            IndividualPkmCounter.matchup_table[j][i] = wins[1] / self.n_battles

            average_winrate = np.sum(IndividualPkmCounter.matchup_table, axis=1) / IndividualPkmCounter.n_pkms
            self.policy = softmax(average_winrate)

    def select_top_tier_indices(self, top_k=10):
        # 여기서 임의의 기준을 사용하여 상위 티어 포켓몬의 인덱스를 선택합니다.
        # 예를 들어, 초기 평균 성능, 이전 기록 등을 기준으로 할 수 있습니다.
        # 현재는 단순히 상위 10개의 포켓몬을 선택하도록 하겠습니다.
        return np.argsort(np.random.random(IndividualPkmCounter.n_pkms))[:top_k]

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        members: List[int] = np.random.choice(IndividualPkmCounter.n_pkms, 3, False, p=self.policy)
        return PkmFullTeam([IndividualPkmCounter.pkms[members[0]], IndividualPkmCounter.pkms[members[1]], IndividualPkmCounter.pkms[members[2]]])

def select_next(matchup_table, n_pkms, members, coverage_weight, t=0.5, r=0.5):
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

class Real_Good_Team(IndividualPkmCounter):
    def __init__(self):
        super().__init__()

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        coverage_weight = np.array([1.0] * IndividualPkmCounter.n_pkms)
        members: List[int] = []
        matchup_table = deepcopy(IndividualPkmCounter.matchup_table)
        select_next(matchup_table, IndividualPkmCounter.n_pkms, members, coverage_weight)
        return PkmFullTeam([IndividualPkmCounter.pkms[members[0]], IndividualPkmCounter.pkms[members[1]], IndividualPkmCounter.pkms[members[2]]])
