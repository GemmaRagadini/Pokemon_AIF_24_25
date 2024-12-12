from typing import List
import random

from vgc.behaviour import BattlePolicy
from vgc.datatypes.Objects import GameState, PkmMove
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition, PkmStatus
from vgc.datatypes.Constants import DEFAULT_N_ACTIONS, DEFAULT_PKM_N_MOVES, TYPE_CHART_MULTIPLIER, DEFAULT_PARTY_SIZE

# 포켓몬 기술의 예상 데미지를 계산하는 함수
def estimate_damage(move: PkmMove, attacker: PkmType, defender: PkmType,
                    attack_stage: int, defense_stage: int, weather: WeatherCondition, attacker_status: PkmStatus) -> float:
    move_type = move.type
    move_power = move.power
    stab = 1.5 if move_type == attacker else 1.
    if (move_type == PkmType.WATER and weather == WeatherCondition.RAIN) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.SUNNY):
        weather_bonus = 1.5
    elif (move_type == PkmType.WATER and weather == WeatherCondition.SUNNY) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.RAIN):
        weather_bonus = .5
    else:
        weather_bonus = 1.
    stage_level = attack_stage - defense_stage
    stage = (stage_level + 2.) / 2 if stage_level >= 0. else 2. / (abs(stage_level) + 2.)
    
    # 상태 이상에 따른 데미지 감소
    if attacker_status == PkmStatus.BURNED :
        stage *= 0.5
    elif attacker_status == PkmStatus.PARALYZED:
        stage *= 0.75
    
    damage = TYPE_CHART_MULTIPLIER[move_type][defender] * stab * weather_bonus * stage * move_power
    return damage

# 포켓몬 간의 상성을 평가
def evaluate_matchup(pkm_type: PkmType, opp_pkm_type: PkmType, moves_type: List[PkmType]) -> float:
    effectiveness = 1.0
    for mtype in moves_type:
        effectiveness *= TYPE_CHART_MULTIPLIER[mtype][pkm_type]
    return effectiveness

class EnhancedBattlePolicy(BattlePolicy):
    def __init__(self, switch_probability: float = .15, n_moves: int = DEFAULT_PKM_N_MOVES,
                 n_switches: int = DEFAULT_PARTY_SIZE):
        super().__init__()
        self.opp_stats_history = []  # 상대 포켓몬의 능력치 변화 추적

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def is_status_move(self, move) -> bool:
        return move.power == 0 and move.effect is not None

    def get_action(self, g: GameState) -> int:
        weather = g.weather.condition

        my_team = g.teams[0]
        my_active = my_team.active
        my_party = my_team.party
        my_attack_stage = my_team.stage[PkmStat.ATTACK]
        my_defense_stage = my_team.stage[PkmStat.DEFENSE]
        my_status = my_active.status

        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_attack_stage = opp_team.stage[PkmStat.ATTACK]
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        # 상대 포켓몬의 능력치 변화 추적
        self.opp_stats_history.append((opp_active, opp_attack_stage, opp_defense_stage))

        # 1. 상태 이상 기술을 사용할 수 있는지 확인
        for i, move in enumerate(my_active.moves):
            if self.is_status_move(move):
                return i

        # 2. 상대 포켓몬에게 큰 피해를 줄 수 있는 기술을 사용
        best_damage = 0
        best_move = 0
        for i, move in enumerate(my_active.moves):
            damage = estimate_damage(move, my_active.type, opp_active.type, my_attack_stage,
                                     opp_defense_stage, weather, my_status)
            if damage > best_damage:
                best_damage = damage
                best_move = i
        
        # 3. 효과가 좋은 기술로 공격할 수 없는 경우, 교체 고려
        if best_damage == 0:
            best_switch = self.find_best_switch(my_party, opp_active)
            if best_switch is not None:
                return best_switch + 4

        # 4. 상대의 주요 기술의 PP를 소진시키는 전략을 사용
        for i, move in enumerate(my_active.moves):
            if move.pp < 5 and move.pp > 0:
                return i

        # 5. 예측 기반 방어 전략을 사용.
        predicted_move = random.choice(opp_active.moves)
        if predicted_move.power > 0 and TYPE_CHART_MULTIPLIER[predicted_move.type][my_active.type] > 1.0:
            for i, pkm in enumerate(my_party):
                if not pkm.fainted() and TYPE_CHART_MULTIPLIER[predicted_move.type][pkm.type] < 1.0:
                    return DEFAULT_PKM_N_MOVES + i

        return best_move


    
    def find_best_switch(self, my_party, opp_active):
        best_matchup = 0
        best_switch = None
        for i, pkm in enumerate(my_party):
            if not pkm.fainted():
                matchup = evaluate_matchup(pkm.type, opp_active.type, list(map(lambda m: m.type, opp_active.moves)))
                if matchup > best_matchup:
                    best_matchup = matchup
                    best_switch = i
        return best_switch
    
# References:
# The following code sections were referenced from the winner's code of vgc 2023 DominikBaziukCompetitor - Dominik Baziuk
# URL or other relevant details (e.g.,https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/tree/master/competition/vgc2023/DominikBaziukCompetitor%20%20-%20Dominik%20Baziuk?ref_type=heads)