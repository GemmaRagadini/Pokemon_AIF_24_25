from typing import List

import numpy as np

from vgc.behaviour import BattlePolicy
from vgc.competition.Competitor import Competitor
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER
from vgc.datatypes.Objects import GameState
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition, PkmEntryHazard


# Eval functions
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


def evaluate_matchup(pkm_type: PkmType, opp_pkm_type: PkmType, moves_type: List[PkmType]) -> float:
    # determine defensive matchup
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


# My Battle Policy
class DBaziukBattlePolicy(BattlePolicy):
    def __init__(self, switch_probability: float = .15, n_moves: int = DEFAULT_PKM_N_MOVES,
                 n_switches: int = DEFAULT_PARTY_SIZE):
        super().__init__()
        self.hail_used = False
        self.sandstorm_used = False

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g: GameState) -> int:
        # print("It's me")
        # get weather condition
        weather = g.weather.condition

        # get my pkms
        my_team = g.teams[0]
        my_active = my_team.active
        my_party = my_team.party
        my_attack_stage = my_team.stage[PkmStat.ATTACK]
        my_defense_stage = my_team.stage[PkmStat.DEFENSE]

        # get opp team
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_not_fainted_pkms = len(opp_team.get_not_fainted())
        opp_attack_stage = opp_team.stage[PkmStat.ATTACK]
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        # estimate damage pkm moves
        damage: List[float] = []
        for move in my_active.moves:
            damage.append(estimate_damage(move.type, my_active.type, move.power, opp_active.type, my_attack_stage,
                                          opp_defense_stage, weather))

        # get most damaging move
        move_id = int(np.argmax(damage))

        #  If this damage is greater than the opponents current health we knock it out
        if damage[move_id] >= opp_active.hp:
            # print("try to knock it out")
            return move_id

        # If move is super effective use it
        if damage[move_id] > 0 and TYPE_CHART_MULTIPLIER[my_active.moves[move_id].type][opp_active.type] == 2.0:
            # print("Attack with supereffective")
            return move_id

        defense_type_multiplier = evaluate_matchup(my_active.type, opp_active.type,
                                                   list(map(lambda m: m.type, opp_active.moves)))
        # print(defense_type_multiplier)
        if defense_type_multiplier <= 1.0:
            # Check for spike moves if spikes not setted
            if opp_team.entry_hazard != PkmEntryHazard.SPIKES and opp_not_fainted_pkms > DEFAULT_PARTY_SIZE / 2:
                for i in range(DEFAULT_PKM_N_MOVES):
                    if my_active.moves[i].hazard == PkmEntryHazard.SPIKES:
                        # print("Setting Spikes")
                        return i

            # Use sandstorm if not setted and you have pokemons immune to that
            if weather != WeatherCondition.SANDSTORM and not self.sandstorm_used and defense_type_multiplier < 1.0:
                sandstorm_move = -1
                for i in range(DEFAULT_PKM_N_MOVES):
                    if my_active.moves[i].weather == WeatherCondition.SANDSTORM:
                        sandstorm_move = i
                immune_pkms = 0
                for pkm in my_party:
                    if not pkm.fainted() and pkm.type in [PkmType.GROUND, PkmType.STEEL, PkmType.ROCK]:
                        immune_pkms += 1
                if sandstorm_move != -1 and immune_pkms > 2:
                    # print("Using Sandstorm")
                    self.sandstorm_used = True
                    return sandstorm_move

            # Use hail if not setted and you have pokemons immune to that
            if weather != WeatherCondition.HAIL and not self.hail_used and defense_type_multiplier < 1.0:
                hail_move = -1
                for i in range(DEFAULT_PKM_N_MOVES):
                    if my_active.moves[i].weather == WeatherCondition.HAIL:
                        hail_move = i
                immune_pkms = 0
                for pkm in my_party:
                    if not pkm.fainted() and pkm.type in [PkmType.ICE]:
                        immune_pkms += 1
                if hail_move != -1 and immune_pkms > 2:
                    # print("Using Hail")
                    self.hail_used = True
                    return hail_move

            # If enemy attack and defense stage is 0 , try to use attack or defense down
            if opp_attack_stage == 0 and opp_defense_stage == 0:
                for i in range(DEFAULT_PKM_N_MOVES):
                    if my_active.moves[i].target == 1 and my_active.moves[i].stage != 0 and (
                            my_active.moves[i].stat == PkmStat.ATTACK or my_active.moves[i].stat == PkmStat.DEFENSE):
                        # print("Debuffing enemy")
                        return i

            # If spikes not set try to switch
            # print("Attacking enemy to lower his hp")
            return move_id

        # If we are not switch, find pokemon with resistance 
        matchup: List[float] = []
        not_fainted = False
        for pkm in my_party:
            if pkm.hp == 0.0:
                matchup.append(0.0)
            else:
                not_fainted = True
                matchup.append(
                    evaluate_matchup(pkm.type, opp_active.type, list(map(lambda m: m.type, opp_active.moves))))

        best_switch = int(np.argmin(matchup))
        if not_fainted and my_party[best_switch] != my_active:
            # print("Switching to someone else")
            return best_switch + 4

        # If our party has no non fainted pkm, lets give maximum possible damage with current active
        # print("Nothing to do just attack")
        return move_id


class MyCompetitor(Competitor):

    def __init__(self, name: str = "Dominik Baziuk (Ocean Man)"):
        self._name = name
        self._battle_policy = DBaziukBattlePolicy()

    @property
    def name(self):
        return self._name

    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy
