import numpy as np
from vgc.datatypes.Objects import PkmStatus, WeatherCondition, GameState
from vgc.datatypes.Types import PkmType, PkmStat
from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER
from typing import List


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
    stage = (stage_level + 2.) / 2 if stage_level >= 0. else 2. / (np.abs(stage_level) + 2.)
    damage = TYPE_CHART_MULTIPLIER[move_type][opp_pkm_type] * stab * weather * stage * move_power
    return damage


def match_up_eval(my_pkm_type: PkmType, opp_pkm_type: PkmType, opp_moves_type: List[PkmType]) -> float:
    defensive_match_up = 0.0
    for mtype in opp_moves_type + [opp_pkm_type]:
        defensive_match_up = max(TYPE_CHART_MULTIPLIER[mtype][my_pkm_type], defensive_match_up)
    return defensive_match_up


class MyBattlePolicy(BattlePolicy):
    def __init__(self, switch_probability: float = 0.2, n_moves: int = DEFAULT_PKM_N_MOVES,
                 n_switches: int = DEFAULT_PARTY_SIZE):
        super().__init__()
        self.switch_probability = switch_probability
        self.n_moves = n_moves
        self.n_switches = n_switches
        self.hail_used = False
        self.sandstorm_used = False
        self.last_switch = None

    def get_action(self, g):
        weather = g.weather.condition if g.weather else WeatherCondition.NORMAL

        my_team = g.teams[0]
        my_active = my_team.active
        my_party = my_team.party
        my_attack_stage = my_team.stage[PkmStat.ATTACK]
        my_defense_stage = my_team.stage[PkmStat.DEFENSE]

        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_attack_stage = opp_team.stage[PkmStat.ATTACK]
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]
        
        for i, move in enumerate(my_active.moves):
            if move and move.power >= 100:
                return i

        if self.should_switch_on_status(my_active):
            best_switch = self.find_best_switch(my_team, opp_active)
            if best_switch is not None and (self.last_switch is None or best_switch.type != self.last_switch.type):
                self.last_switch = best_switch
                return my_team.party.index(best_switch)

        special_move_id = self.select_special_move(my_active, my_party, opp_team.party, weather)
        if special_move_id != -1:
            return special_move_id

        best_move_id = self.find_best_move(my_active, opp_active, my_attack_stage, opp_defense_stage, weather)

        if self.should_switch(my_team, opp_active):
            best_switch = self.find_best_switch(my_team, opp_active)
            if best_switch is not None and (self.last_switch is None or best_switch.type != self.last_switch.type):
                self.last_switch = best_switch
                return my_team.party.index(best_switch)

        self.last_switch = None
        return best_move_id

    def find_best_move(self, my_pkm, opp_pkm, my_attack_stage, opp_defense_stage, weather):
        move_scores = {}
        for i, move in enumerate(my_pkm.moves):
            dmg = estimate_damage(move.type, my_pkm.type, move.power, opp_pkm.type, my_attack_stage, opp_defense_stage,
                                  weather)
            type_effect = TYPE_CHART_MULTIPLIER[move.type][opp_pkm.type]
            accuracy = move.acc
            score = dmg * type_effect * accuracy
            if opp_pkm.hp <= dmg:
                score *= 1.5
            move_scores[i] = score
        return max(move_scores, key=move_scores.get)

    def find_best_switch(self, my_team, opp_pkm):
        best_score = float('-inf')
        best_switch = None
        for pkm in my_team.party:
            if not pkm.fainted() and pkm != my_team.active:
                score = match_up_eval(pkm.type, opp_pkm.type, [move.type for move in pkm.moves])
                if pkm.hp / pkm.max_hp > 0.5:
                    score *= 1.2
                if score > best_score:
                    best_score = score
                    best_switch = pkm
        return best_switch

    def should_switch_on_status(self, my_pkm):
        return my_pkm.status in [PkmStatus.PARALYZED, PkmStatus.POISONED, PkmStatus.CONFUSED, PkmStatus.BURNED,
                                 PkmStatus.FROZEN, PkmStatus.SLEEP]

    def should_switch(self, my_team, opp_pkm):
        return np.random.rand() < self.switch_probability

    def select_special_move(self, my_pkm, my_party, opp_party, weather):
        if weather != WeatherCondition.SANDSTORM and not self.sandstorm_used:
            sandstorm_move = self.find_move_by_weather(my_pkm, WeatherCondition.SANDSTORM)
            if sandstorm_move != -1:
                opp_weak = self.count_pkms_weak_to_sandstorm(opp_party)
                my_immune = self.count_pkms_immune_to_sandstorm(my_party)
                if opp_weak >= 2 and my_immune >= 2:
                    self.sandstorm_used = True
                    return sandstorm_move

        if weather != WeatherCondition.HAIL and not self.hail_used:
            hail_move = self.find_move_by_weather(my_pkm, WeatherCondition.HAIL)
            if hail_move != -1:
                opp_weak = self.count_pkms_weak_to_hail(opp_party)
                my_immune = self.count_pkms_immune_to_hail(my_party)
                if opp_weak >= 2 and my_immune >= 2:
                    self.hail_used = True
                    return hail_move

        return -1

    def find_move_by_weather(self, my_pkm, weather):
        for i, move in enumerate(my_pkm.moves):
            if move and move.weather == weather:
                return i
        return -1

    def count_pkms_weak_to_sandstorm(self, party):
        return sum(1 for pkm in party if pkm.type in [PkmType.FIRE, PkmType.ELECTRIC, PkmType.POISON])

    def count_pkms_immune_to_sandstorm(self, party):
        return sum(1 for pkm in party if pkm.type in [PkmType.GROUND, PkmType.STEEL, PkmType.ROCK])

    def count_pkms_weak_to_hail(self, party):
        return sum(1 for pkm in party if pkm.type in [PkmType.GROUND, PkmType.FLYING, PkmType.DRAGON])

    def count_pkms_immune_to_hail(self, party):
        return sum(1 for pkm in party if pkm.type == PkmType.ICE)
