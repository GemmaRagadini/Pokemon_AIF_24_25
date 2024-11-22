from typing import List

import numpy as np
import random

from vgc.behaviour import BattlePolicy
from vgc.competition.Competitor import Competitor

from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER
from vgc.datatypes.Objects import PkmMove, GameState, Pkm, PkmTeam
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition, PkmStatus


# Matching attacker to defender
def match_up_eval(defender_type: PkmType, attacker_type: PkmType, attacker_moves_type: List[PkmType]) -> float:
    # determine offensive match up
    offensive_match_up = 0.0
    for mtype in attacker_moves_type + [attacker_type]:
        offensive_match_up = max(TYPE_CHART_MULTIPLIER[mtype][defender_type], offensive_match_up)
    return offensive_match_up


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


def find_super_attack_pokemon(t: PkmTeam, opp: Pkm, move_index) -> int:
    best_index = None

    # Do any of the pokemon in our party have 2.0 effects against opponent's pokemon?
    for i, pkm in enumerate(t.party):
        if pkm.fainted():
            continue
        if match_up_eval(opp.type, pkm.type, list(map(lambda m: m.type, pkm.moves))) == 2.0:
            best_index = i + 4
    return best_index if best_index is not None else move_index


def find_super_defense_pokemon(t: PkmTeam, opp: Pkm, move_index) -> int:
    best_index = None

    # Do any of the pokemon in our party opponent have 0.5, 1.0 effects
    for i, pkm in enumerate(t.party):
        if pkm.fainted():
            continue
        if match_up_eval(pkm.type, opp.type, list(map(lambda m: m.type, opp.moves))) == 0.5:
            best_index = i + 4
            break
        elif match_up_eval(pkm.type, opp.type, list(map(lambda m: m.type, opp.moves))) == 1.0:
            best_index = i + 4
    return best_index if best_index is not None else move_index


class WonjunBattlePolicy(BattlePolicy):

    def get_action(self, g: GameState):
        # get weather condition
        weather = g.weather.condition

        # get my pkms
        my_team = g.teams[0]
        my_active = my_team.active
        my_attack_stage = my_team.stage[PkmStat.ATTACK]

        # get opp team
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_active_type = opp_active.type
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        # get most damaging move from my active pkm
        damage: List[float] = []
        for move in my_active.moves:
            damage.append(estimate_damage(move.type, my_active.type, move.power, opp_active_type,
                                          my_attack_stage, opp_defense_stage, weather))

        move_index = int(np.argmax(damage))
        opp_attack_on_me = match_up_eval(my_active.type, opp_active.type, list(map(lambda m: m.type, opp_active.moves)))
        my_attack_on_opp = match_up_eval(opp_active.type, my_active.type, list(map(lambda m: m.type, my_active.moves)))

        # Is a finishing attack possible?
        if damage[move_index] >= opp_active.hp:
            return move_index

        # Check if Spore is available and the opponent is not asleep
        sleep_index = None
        for i, move in enumerate(my_active.moves):
            if move.name == "Spore":
                sleep_index = i
                break
        if sleep_index is not None and opp_active.status == PkmStatus.NONE:
            return sleep_index

        # Check if the currently active pokemon has a 2.0 effect on the opponent's pokemon
        if my_attack_on_opp == 2.0:
            return move_index

        # if not, If my party has a pokemon with a 2.0 effect, trade it.
        super_change_index = find_super_attack_pokemon(my_team, opp_active, move_index)
        if super_change_index > 3:
            return super_change_index  # switch 4 or 5 -> first party pokemon, second party pokemon

        # if opponent has a 2.0 effect on me and I have a 0.5 effect on opponent.
        if opp_attack_on_me == 2.0 and my_attack_on_opp == 0.5:
            super_change_index = find_super_defense_pokemon(my_team, opp_active, move_index)

        # find an effective defense pokemon
        if super_change_index > 3:
            return super_change_index  # switch 4 or 5 -> first party pokemon, second party pokemon

        # Check for status moves with prob >= 0.3 and if the opponent has no status condition
        # Exclude specific status conditions
        excluded_statuses = [PkmStatus.POISONED]
        status_index: List[int] = []

        if opp_active.status == PkmStatus.NONE:
            status_index = [
                i for i, move in enumerate(my_active.moves)
                if move.prob >= 0.3 and move.status is not None and move.status not in excluded_statuses
            ]
        if status_index:
            return random.choice(status_index)

        return move_index


class WonjunCompetitor(Competitor):

    def __init__(self, name: str = "Wonjun"):
        self._name = name
        self._battle_policy = WonjunBattlePolicy()

    @property
    def name(self):
        return self._name

    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy
