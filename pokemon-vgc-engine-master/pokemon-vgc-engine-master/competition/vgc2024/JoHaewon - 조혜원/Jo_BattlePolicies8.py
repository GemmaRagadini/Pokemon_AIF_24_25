import numpy as np
import random
from typing import List, Tuple
from copy import deepcopy
from functools import lru_cache

from vgc.behaviour import BattlePolicy, TeamSelectionPolicy, TeamBuildPolicy
from vgc.competition.Competitor import Competitor
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER
from vgc.datatypes.Objects import GameState, Pkm
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition, PkmEntryHazard, PkmStatus


@lru_cache(maxsize=None)
def estimate_damage(move_type: PkmType, pkm_type: PkmType, move_power: float, opp_pkm_type: PkmType,
                    attack_stage: int, defense_stage: int, weather: WeatherCondition) -> float:
    stab = 1.5 if move_type == pkm_type else 1.0
    if (move_type == PkmType.WATER and weather == WeatherCondition.RAIN) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.SUNNY):
        weather_modifier = 1.5
    elif (move_type == PkmType.WATER and weather == WeatherCondition.SUNNY) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.RAIN):
        weather_modifier = 0.5
    else:
        weather_modifier = 1.0

    stage_level = attack_stage - defense_stage
    if stage_level >= 0:
        stage_modifier = (stage_level + 2.0) / 2
    else:
        stage_modifier = 2.0 / (abs(stage_level) + 2.0)

    damage = TYPE_CHART_MULTIPLIER[move_type][opp_pkm_type] * stab * weather_modifier * stage_modifier * move_power
    return damage

@lru_cache(maxsize=None)
def evaluate_matchup(pkm_type: PkmType, opp_pkm_type: PkmType, moves_type: Tuple[PkmType, ...]) -> float:
    double_damage = False
    normal_damage = False
    half_damage = False

    for mtype in moves_type:
        damage = TYPE_CHART_MULTIPLIER[mtype][pkm_type]
        if damage == 2.0:
            double_damage = True
        elif damage == 1.0:
            normal_damage = True
        elif damage == 0.5:
            half_damage = True

    if double_damage:
        return 2.0

    return TYPE_CHART_MULTIPLIER[opp_pkm_type][pkm_type]

class haewon_battlepolicies(BattlePolicy):

    def __init__(self, switch_probability: float = 0.15, n_moves: int = DEFAULT_PKM_N_MOVES,
                 n_switches: int = DEFAULT_PARTY_SIZE, depth: int = 2):
        super().__init__()
        self.hail_used = False
        self.sandstorm_used = False
        self.depth = depth

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def evaluate_state(self, g: GameState) -> float:
        my_team = g.teams[0]
        opp_team = g.teams[1]

        my_score = sum(pkm.hp for pkm in [my_team.active] + my_team.party if pkm.hp > 0)
        opp_score = sum(pkm.hp for pkm in [opp_team.active] + opp_team.party if pkm.hp > 0)

        my_type_advantage = sum(TYPE_CHART_MULTIPLIER[pkm.type][opp_team.active.type] for pkm in [my_team.active] + my_team.party)
        opp_type_advantage = sum(TYPE_CHART_MULTIPLIER[opp_team.active.type][pkm.type] for pkm in [my_team.active] + my_team.party)

        my_status_penalty = sum(pkm.status != PkmStatus.NONE for pkm in [my_team.active] + my_team.party)
        opp_status_penalty = sum(pkm.status != PkmStatus.NONE for pkm in [opp_team.active] + opp_team.party)

        return (my_score - opp_score) + (my_type_advantage - opp_type_advantage) - (my_status_penalty - opp_status_penalty)

    def minimax(self, g: GameState, depth: int, alpha: float, beta: float, maximizing_player: bool) -> Tuple[float, int]:
        if depth == 0 or g.teams[0].fainted() or g.teams[1].fainted():
            return self.evaluate_state(g), -1

        if maximizing_player:
            max_eval = float('-inf')
            best_move = -1
            for move_id in range(DEFAULT_PKM_N_MOVES + DEFAULT_PARTY_SIZE):
                next_state = self.simulate_move(g, move_id)
                eval, _ = self.minimax(next_state, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move_id
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = -1
            for move_id in range(DEFAULT_PKM_N_MOVES + DEFAULT_PARTY_SIZE):
                next_state = self.simulate_move(g, move_id)
                eval, _ = self.minimax(next_state, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move_id
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def simulate_move(self, g: GameState, move_id: int) -> GameState:
        new_g = deepcopy(g)
        my_team = new_g.teams[0]
        opp_team = new_g.teams[1]
        my_active = my_team.active
        opp_active = opp_team.active

        if move_id < DEFAULT_PKM_N_MOVES:
            move = my_active.moves[move_id]
            damage = estimate_damage(move.type, my_active.type, move.power, opp_active.type,
                                     my_team.stage[PkmStat.ATTACK], opp_team.stage[PkmStat.DEFENSE], new_g.weather.condition)
            opp_active.hp = max(0, opp_active.hp - damage)
        else:
            switch_id = move_id - DEFAULT_PKM_N_MOVES
            if switch_id < len(my_team.party):
                my_team.active, my_team.party[switch_id] = my_team.party[switch_id], my_team.active

        return new_g

    def get_best_attack_option(self, damage: List[float], my_active: Pkm, opp_active: Pkm) -> int:
        effective_damage = [dmg if dmg < opp_active.hp else opp_active.hp for dmg in damage]
        move_id = int(np.argmax(effective_damage))

        for idx, move in enumerate(my_active.moves):
            if move.prob > 0 and (move.status != PkmStatus.NONE or move.stage != 0 or move.recover > 0 or move.fixed_damage > 0 or move.weather != WeatherCondition.CLEAR or move.hazard != PkmEntryHazard.NONE):
                if effective_damage[idx] > opp_active.hp / 2:
                    move_id = idx
                    break
        
        return move_id

    def get_action(self, g: GameState) -> int:
        _, best_move = self.minimax(g, self.depth, float('-inf'), float('inf'), True)
        return best_move
