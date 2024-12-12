import numpy as np

from pygad import pygad
from scipy.optimize import linprog
from torch import nn

from vgc.balance.meta import MetaData
from vgc.behaviour import TeamBuildPolicy, BattlePolicy
from vgc.behaviour.BattlePolicies import TypeSelector
from vgc.competition.StandardPkmMoves import STANDARD_MOVE_ROSTER
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, MAX_HIT_POINTS
from vgc.datatypes.Objects import Pkm, PkmTemplate, PkmFullTeam, PkmRoster, PkmTeam, PkmMove
from vgc.datatypes.Types import N_TYPES, N_STATUS, N_ENTRY_HAZARD, PkmStat
from vgc.engine.PkmBattleEnv import PkmBattleEnv


class LucyBuildPolicy(TeamBuildPolicy):
    def __init__(self):
        self.roster = None
        self.stat_weights = {
            'HP': 0.4,
            'POWER': 0.2,
            'PRIORITY': 0.1,
            'ACCURACY': 0.1,
            'STAT_CHANGE': 0.2
        }

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        self.roster = roster

    def calculate_score(self, pkm: Pkm) -> float:
        move_scores = [
            self.stat_weights['POWER'] * move.power +
            self.stat_weights['PRIORITY'] * move.priority +
            self.stat_weights['ACCURACY'] * move.acc +
            self.stat_weights['STAT_CHANGE'] * move.stage
            for move in pkm.moves
        ]
        score = (
            pkm.max_hp * self.stat_weights['HP'] +
            sum(move_scores) / len(pkm.moves)  # 평균 move score 계산
        )
        return score

    def get_action(self, meta) -> PkmFullTeam:
        selected_pokemons = []
        selected_types = set()
        type_to_best_pkm = {}

        # 각 타입별로 스탯 점수가 가장 높은 포켓몬을 선택
        for pt in self.roster:
            pkm = pt.gen_pkm([0, 1, 2, 3])  # 임의의 기술 선택
            score = self.calculate_score(pkm)
            if pkm.type not in type_to_best_pkm or score > type_to_best_pkm[pkm.type][1]:
                type_to_best_pkm[pkm.type] = (pkm, score)

        # 타입별로 한 포켓몬씩 선택
        for pkm, score in type_to_best_pkm.values():
            if len(selected_pokemons) >= 3:
                break
            selected_pokemons.append(pkm)
            selected_types.add(pkm.type)

        # 선택된 포켓몬이 3개가 되지 않았을 경우, 남은 포켓몬 중에서 스탯 점수가 높은 순으로 추가 선택
        remaining_pokemons = [
            (pt.gen_pkm([0, 1, 2, 3]), self.calculate_score(pt.gen_pkm([0, 1, 2, 3])))
            for pt in self.roster
            if pt.gen_pkm([0, 1, 2, 3]).type not in selected_types
        ]
        remaining_pokemons.sort(key=lambda x: x[1], reverse=True)
        while len(selected_pokemons) < 3:
            selected_pokemons.append(remaining_pokemons.pop(0)[0])

        return PkmFullTeam(selected_pokemons)