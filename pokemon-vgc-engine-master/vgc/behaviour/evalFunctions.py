from vgc.datatypes.Types import PkmType, WeatherCondition
from typing import List
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER
from vgc.datatypes.Objects import GameState, PkmTeam, PkmStat, Pkm
import numpy as np

"""
Evaluation functions 
"""

def game_state_eval(s: GameState, depth):
    """
    Pre-existing evaluation function
    """
    mine = s.teams[0].active
    opp = s.teams[1].active
    return mine.hp / mine.max_hp - 3 * opp.hp / opp.max_hp - 0.3 * depth

def estimate_damage(move_type: PkmType, pkm_type: PkmType, move_power: float, opp_pkm_type: PkmType,
                    attack_stage: int, defense_stage: int, weather: WeatherCondition) -> float:
    """
    (pre-existing)
    Estimate the damage that the move moveType deals to the opposing Pokémon, taking into account:
    - types of both Pokémon
    - type of the move
    - attack and defense stats
    - weather
    - move power
    """
    # Bonus for same-type move used by the Pokémon
    stab = 1.5 if move_type == pkm_type else 1.
    # Favorable weather
    if (move_type == PkmType.WATER and weather == WeatherCondition.RAIN) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.SUNNY):
        weather = 1.5
    # Unfavorable weather
    elif (move_type == PkmType.WATER and weather == WeatherCondition.SUNNY) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.RAIN):
        weather = .5
    else:
        weather = 1.
    # relative level attack - defense 
    stage_level = attack_stage - defense_stage # in [-10,10]
    # Multiplier that increases linearly for positive values and applies fractional reductions for negative values
    stage = (stage_level + 2.) / 2 if stage_level >= 0. else 2. / (np.abs(stage_level) + 2.) # in [0,6]
    # damage estimation
    damage = TYPE_CHART_MULTIPLIER[move_type][opp_pkm_type] * stab * weather * stage * move_power # Approximately 140
    return damage



def my_eval_fun(s:GameState, depth):
    """
    Evaluation function that considers the compatibility between Pokémon, 
    the game_state_eval with respect to hp, and the ability to deal damage
    """
    my_active = s.teams[0].active
    opp_active = s.teams[1].active
    attack_stage = s.teams[0].stage[PkmStat.ATTACK]
    defense_stage = s.teams[1].stage[PkmStat.DEFENSE]
    matchup = examine_matchup(my_active.type, opp_active.type, list(map(lambda m: m.type, my_active.moves))) # in [0,2]
    eval_hp = game_state_eval(s,depth) + 4 # in [0-5]
    max_damage = maxDamage(my_active, opp_active.type, attack_stage, defense_stage, s.weather) # in [0,140]
    return max_damage/70 + matchup/2 + eval_hp


def maxDamage(my_active: Pkm, opp_active_type:PkmType, attack_stage: int, defense_stage: int,weather: WeatherCondition ): 
    """
    Returns the maximum damage the active Pokémon can deal to the opponent with a move
    """
    mvs_damage = [] 
    # estimate the damage for each move of my Pokémon
    for m in my_active.moves:
        mvs_damage.append(estimate_damage(m.type,my_active.type, m.power, opp_active_type ,attack_stage, defense_stage, weather))
    return np.max(mvs_damage)


def examine_matchup(pkm_type: PkmType, opp_pkm_type: PkmType, moves_type: List[PkmType]) -> float:
    """ 
    Evaluates the matchup between the active Pokémon and the opponent's Pokémon, 
    considering the types of the Pokémon and the available moves.
    """
    for mtype in moves_type: # search for super effective move 
        if TYPE_CHART_MULTIPLIER[mtype][pkm_type] == 2.0:
            return 2.0  # if there is a super effective move
    # Consider only the evaluation based on the Pokémon's type
    return TYPE_CHART_MULTIPLIER[opp_pkm_type][pkm_type]



def n_defeated(t: PkmTeam):
    """
    compute the number of defeated pokemon in the team
    """
    fainted = 0
    fainted += t.active.hp == 0
    if len(t.party) > 0:
        fainted += t.party[0].hp == 0
    if len(t.party) > 1:
        fainted += t.party[1].hp == 0
    return fainted