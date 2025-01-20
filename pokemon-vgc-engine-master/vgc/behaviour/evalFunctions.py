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
    Funzione di valutazione preesistente 
    """
    mine = s.teams[0].active
    opp = s.teams[1].active
    return mine.hp / mine.max_hp - 3 * opp.hp / opp.max_hp - 0.3 * depth

def estimate_damage(move_type: PkmType, pkm_type: PkmType, move_power: float, opp_pkm_type: PkmType,
                    attack_stage: int, defense_stage: int, weather: WeatherCondition) -> float:
    """
    (preesistente)
    Stima il danno che la mossa moveType infligge al pokemon avversario, tenendo conto di:
    - tipi dei due pokemon
    - tipo della mossa
    - statistiche di attacco difesa
    - meteo 
    - potenza della mossa 
    """
    # bonus per mossa di stesso tipo del pokemon che la usa 
    stab = 1.5 if move_type == pkm_type else 1.
    # condizione meteo favorevole 
    if (move_type == PkmType.WATER and weather == WeatherCondition.RAIN) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.SUNNY):
        weather = 1.5
    # condizione meteo sfavorevole
    elif (move_type == PkmType.WATER and weather == WeatherCondition.SUNNY) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.RAIN):
        weather = .5
    else:
        weather = 1.
    # livello relativo attacco - difesa 
    stage_level = attack_stage - defense_stage # in [-10,10]
    # moltiplicatore che cresce linearmente per valori positivi e riduzioni frazionali per valori negativi
    stage = (stage_level + 2.) / 2 if stage_level >= 0. else 2. / (np.abs(stage_level) + 2.) # in [0,6]
    # stima del danno 
    damage = TYPE_CHART_MULTIPLIER[move_type][opp_pkm_type] * stab * weather * stage * move_power # circa [0,4000]
    return damage



def my_eval_fun(s:GameState, depth):
    """
    Funzione di valutazione che considera la compatibilità tra i pokemon, la game_state_eval rispetto agli hp e 
    la possibilità di infliggere danno
    """
    my_active = s.teams[0].active
    opp_active = s.teams[1].active
    attack_stage = s.teams[0].stage[PkmStat.ATTACK]
    defense_stage = s.teams[1].stage[PkmStat.DEFENSE]
    matchup = evaluate_matchup(my_active.type, opp_active.type, list(map(lambda m: m.type, my_active.moves))) # in [0,2]
    eval_hp = game_state_eval(s,depth) + 4 # circa in [0-5]
    max_damage = maxDamage(my_active, opp_active.type, attack_stage, defense_stage, s.weather) # in [0,140]
    return max_damage/140 + matchup/2 + eval_hp


def maxDamage(my_active: Pkm, opp_active_type:PkmType, attack_stage: int, defense_stage: int,weather: WeatherCondition ): 
    """
    Ritorna il massimo danno il pokemon attivo poù infliggere all'avversario con una mossa
    """
    mvs_damage = [] 
    # stimo il danno per ogni mossa del mio pokemon
    for m in my_active.moves:
        mvs_damage.append(estimate_damage(m.type,my_active.type, m.power, opp_active_type ,attack_stage, defense_stage, weather))
    return np.max(mvs_damage)
 

def evaluate_matchup(pkm_type: PkmType, opp_pkm_type: PkmType, moves_type: List[PkmType]) -> float:
    """ 
    Valuta l'abbinamento tra il pokemon attivo e il pokemon avversario, 
    considerando i tipi dei pokemon e delle mosse disponibili.
    """
    for mtype in moves_type: # cerca mossa super efficace 
        if TYPE_CHART_MULTIPLIER[mtype][pkm_type] == 2.0:
            return 2.0  # ritorna 2 nel caso in cui ci sia una mossa super efficace 
    # altrimenti considera solo la valutazione rispetto al tipo di pokemon 
    return TYPE_CHART_MULTIPLIER[opp_pkm_type][pkm_type]



def n_fainted(t: PkmTeam):
    """
    Calcola numero di pokemon esausti nel team
    """
    fainted = 0
    fainted += t.active.hp == 0
    if len(t.party) > 0:
        fainted += t.party[0].hp == 0
    if len(t.party) > 1:
        fainted += t.party[1].hp == 0
    return fainted