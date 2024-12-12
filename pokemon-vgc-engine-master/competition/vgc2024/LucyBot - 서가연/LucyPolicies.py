from typing import Tuple, List, Dict
import heapq
import copy

import numpy as np
from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE
from vgc.datatypes.Objects import GameState
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition


class LucyPolicy(BattlePolicy):
    def __init__(self):
        self.sandstorm_used = False
        self.hail_used = False
        # (시간 제한 이슈)
        self.max_depth = 2  # 탐색 깊이 제한 
        self.top_k = 7 # 상위 k개의 행동만 탐색 
        
    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_name(self) -> str:
        return "LucyBot"

    ## a star 탐색 알고리즘 
    def a_star(self, start_state: GameState) -> Tuple[int, float]:
        # 휴리스틱 함수 정의 
        def heuristic(state: GameState) -> float:
            my_team = state.teams[0]
            opp_team = state.teams[1]
            
            ## HP 차이 
            my_hp = sum(pkm.hp for pkm in my_team.party)
            opp_hp = sum(pkm.hp for pkm in opp_team.party)
            hp_difference = my_hp - opp_hp
            
            ## 공격 스테이지 차이 
            my_attack_stage = my_team.stage[PkmStat.ATTACK]
            opp_attack_stage = opp_team.stage[PkmStat.ATTACK]
            attack_stage_difference = my_attack_stage - opp_attack_stage
            
            ## 방어 스테이지 차이 
            my_defense_stage = my_team.stage[PkmStat.DEFENSE]
            opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]
            defense_stage_difference = my_defense_stage - opp_defense_stage
            
            # 가중치를 주어 계산하기 
            return (0.4 * hp_difference + 
                    0.3 * attack_stage_difference + 
                    0.3 * defense_stage_difference)


        def get_state_key(state: GameState) -> Tuple:
            # 각 팀의 상태, 날씨 조건을 키로 사용 
            my_team = state.teams[0]
            opp_team = state.teams[1]
            my_state = (my_team.active.hp, tuple(pkm.hp for pkm in my_team.party))
            opp_state = (opp_team.active.hp, tuple(pkm.hp for pkm in opp_team.party))
            return (my_state, opp_state, state.weather.condition)

        def get_successors(state: GameState, depth: int) -> List[Tuple[GameState, int, float]]:
            successors = []  ## 다음 상태 
        
            ## 탐색 깊이 제한 
            if depth >= self.max_depth:
                return successors

            my_team = state.teams[0]
            my_active = my_team.active
            opp_team = state.teams[1]
            opp_active = opp_team.active

            for move_id, move in enumerate(my_active.moves):
                if my_active.hp <= 0:
                    continue

                new_state = copy.deepcopy(state)
                ## 데미지 확인 
                damage = estimate_damage(move.type, my_active.type, move.power, opp_active.type,
                                              my_team.stage[PkmStat.ATTACK], opp_team.stage[PkmStat.DEFENSE],
                                              state.weather.condition)
                new_state.teams[1].active.hp -= damage
                cost = -damage # 데미지를 비용으로 사용
                successors.append((new_state, move_id, cost))

            # 상위 k개의 행동만 탐색
            successors.sort(key=lambda x: x[2], reverse=True)  # 높은 데미지 순으로 정렬
            return successors[:self.top_k]

        open_set = []
        heapq.heappush(open_set, (0, start_state, None, 0))  # (우선순위, 상태, 행동, 깊이)
        came_from: Dict[Tuple, Tuple[Tuple, int]] = {}
        g_score = {get_state_key(start_state): 0}  # 시작 상태 비용 
        f_score = {get_state_key(start_state): heuristic(start_state)}  # 휴리스틱 값

        while open_set:
            _, current, action, depth = heapq.heappop(open_set)
            current_key = get_state_key(current)

            ## 상대 포켓몬 체력 0 이하일 경우
            if current.teams[1].active.hp <= 0:  
                return action, g_score[current_key]

            for successor, move_id, cost in get_successors(current, depth):
                successor_key = get_state_key(successor)
                tentative_g_score = g_score[current_key] + cost
                if successor_key not in g_score or tentative_g_score < g_score[successor_key]:
                    came_from[successor_key] = (current_key, move_id)
                    g_score[successor_key] = tentative_g_score
                    f_score[successor_key] = tentative_g_score + heuristic(successor)
                    heapq.heappush(open_set, (f_score[successor_key], successor, move_id, depth + 1))

        return None, float('inf')  ## 도달하지 않게끔 



    def get_action(self, g: GameState) -> int:
        ## 최적 행동 선택
        move_id, _ = self.a_star(g)
        if move_id is not None:
            return move_id
        else:
            return np.random.choice(range(DEFAULT_PKM_N_MOVES))


# 기술의 피해량을 추정하는 함수
def estimate_damage(move_type: PkmType, pkm_type: PkmType, move_power: float, opp_pkm_type: PkmType,
                    attack_stage: int, defense_stage: int, weather: WeatherCondition) -> float:
    stab = 1.5 if move_type == pkm_type else 1.
    if (move_type == PkmType.WATER and weather == WeatherCondition.RAIN) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.SUNNY):
        weather = 1.5
    elif (move_type == PkmType.WATER and weather == WeatherCondition.SUNNY) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.RAIN):
        weather = .5
    else:
        weather = 1.
    stage_level = attack_stage - defense_stage
    stage = (stage_level + 2.) / 2 if stage_level >= 0. else 2. / (abs(stage_level) + 2.)
    damage = TYPE_CHART_MULTIPLIER[move_type][opp_pkm_type] * stab * weather * stage * move_power
    return damage
