from typing import List
from vgc.datatypes.Objects import Pkm, PkmTemplate, PkmRoster, PkmFullTeam

from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from pygad import pygad
from scipy.optimize import linprog
from torch import nn

from vgc.balance.meta import MetaData, StandardMetaData
from vgc.behaviour import TeamBuildPolicy, BattlePolicy
from vgc.behaviour.BattlePolicies import TypeSelector
from vgc.competition.StandardPkmMoves import STANDARD_MOVE_ROSTER
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, MAX_HIT_POINTS
from vgc.datatypes.Objects import Pkm, PkmTemplate, PkmFullTeam, PkmRoster, PkmTeam, PkmMove
from vgc.datatypes.Types import N_TYPES, N_STATUS, N_ENTRY_HAZARD
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.util.Encoding import one_hot


from typing import List
from vgc.datatypes.Objects import Pkm, PkmTemplate, PkmRoster, PkmFullTeam

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

from vgc.behaviour.TeamBuildPolicies import IndividualPkmCounter, softmax







import numpy as np
from copy import deepcopy
from typing import List

class MaxPkmCoverage_UCB(IndividualPkmCounter):
    def __init__(self):
        super().__init__()
        #self.counts = np.zeros(IndividualPkmCounter.n_pkms)  # 각 포켓몬 선택 횟수
        #print(IndividualPkmCounter.n_pkms)
        self.counts = None
        self.total_counts = 0

    def select_next(self, matchup_table, n_pkms, members, coverage_weight):
        if self.total_counts < self.n_pkms:
            # 초기에 각 옵션을 적어도 한 번씩 탐색
            available = list(set(range(n_pkms)) - set(members))
            p = np.random.choice(available)
        else:
            # UCB 값 계산
            ucb_values = coverage_weight + np.sqrt(2 * np.log(self.total_counts) / (self.counts + 1))
            # 이미 선택된 포켓몬 제외
            ucb_values[members] = -np.inf
            p = np.argmax(ucb_values)
        
        members.append(p)
        self.counts[p] += 1
        self.total_counts += 1
        
        # 선택된 포켓몬에 따라 가중치 조정
        if len(members) < 3:
            for i in range(n_pkms):
                if matchup_table[p][i] >= 0.5:  # 임계값 t를 0.5로 설정
                    coverage_weight[i] *= 0.5  # 가중치 감소 비율 r을 0.5로 설정
                matchup_table[p][i] = 0  # 해당 포켓몬의 커버리지를 0으로 설정하여 중복 방지
            self.select_next(matchup_table, n_pkms, members, coverage_weight)

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        if self.counts is None:
            self.counts = np.zeros(IndividualPkmCounter.n_pkms)
        coverage_weight = np.array([1.0] * self.n_pkms)
        members = []
        matchup_table = deepcopy(self.matchup_table)
        self.select_next(matchup_table, self.n_pkms, members, coverage_weight)
        return PkmFullTeam([self.pkms[m] for m in members])






class HighHPBuilder(TeamBuildPolicy):
    """
    Team build policy that selects the top three Pokémon with the highest HP from the roster.
    """
    
    def __init__(self):
        self.roster = None  # PkmRoster instance will be set later

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        """
        Set the roster from which the team will be built.
        """
        self.roster = roster

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        # Filter to get only the Pokémon with the highest HP
        max_hp = max(p.max_hp for p in self.roster)  # Determine the highest HP in the roster
        
        high_hp_pokemon = [p for p in self.roster if p.max_hp == max_hp]  # Filter out all Pokémon with max HP
        
        # Evaluate and select the best 3 Pokémon based on their skills
        best_pokemon = self.select_best_pokemon(high_hp_pokemon)
        selected_pokemon = sorted(best_pokemon, key=lambda x: x['score'], reverse=True)[:3]

        # Generate the Pokémon team from the selected best Pokémon
        team = [p['pokemon'].gen_pkm(list(range(DEFAULT_PKM_N_MOVES))) for p in selected_pokemon]
        return PkmFullTeam(team)
    
    def select_best_pokemon(self, pokemon_list):
        """
        Evaluate and score Pokémon based on move priorities and other special characteristics.
        """
        evaluated_pokemon = []
        for pkm in pokemon_list:
            score = 0
            for move in pkm.moves:
                if move.priority:  # High priority moves
                    score += 3
                if move.power > 100:  # High power moves
                    score += 2
                if move.target == 1 and move.stage < 0:
                    score += (move.stage * -1)
                if move.target == 0 and move.stage > 0:
                    score += move.stage
                if move.recover > 0:
                    score += 1
            print("pkm: ", pkm, " ", "score: ", score)
            evaluated_pokemon.append({'pokemon': pkm, 'score': score})
        return evaluated_pokemon
    


