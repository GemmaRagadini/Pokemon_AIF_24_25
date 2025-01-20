import numpy as np
from vgc.behaviour import evalFunctions
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER
from vgc.datatypes.Types import PkmStat, WeatherCondition
from typing import List
from vgc.datatypes.Objects import GameState, PkmType,Pkm
from vgc.behaviour import BattlePolicy


"""
More effective implemented policy
"""


# My Battle Policy
class MyPolicy(BattlePolicy):

    def __init__(self):
        self.hail_used = False
        self.sandstorm_used = False
        self.name = "My Policy"

    def estimate_damages(self, active_pkm: Pkm, opp_pkm_type: PkmType, attack_stage: int, defense_stage: int, weather: WeatherCondition)-> int:
        # valutazione mosse
        damages: List[float] = []
        for move in active_pkm.moves:
            damages.append(evalFunctions.estimate_damage(move.type, active_pkm.type, move.power, opp_pkm_type, attack_stage,
                                          defense_stage, weather))
        return damages


    def get_action(self, g: GameState) -> int:
        # la mia squadra
        my_team = g.teams[0]
        active_pkm = my_team.active
        bench = my_team.party
        my_attack_stage = my_team.stage[PkmStat.ATTACK]
       
        # squadra avversaria
        opp_team = g.teams[1]
        opp_active_pkm = opp_team.active
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        # meteo 
        weather = g.weather.condition

        try:
            # stima dei danni di ogni mossa
            damages = self.estimate_damages(active_pkm, opp_active_pkm.type, my_attack_stage, opp_defense_stage, weather)
            move_id = int(np.argmax(damages))
        except Exception as e:
            import traceback
            traceback.print_exc()

        # se elimina l'avversario oppure il tipo di mossa è superefficace si usa subito: 
        if (damages[move_id] >= opp_active_pkm.hp) or (damages[move_id] > 0 and TYPE_CHART_MULTIPLIER[active_pkm.moves[move_id].type][opp_active_pkm.type] == 2.0) :
            return move_id
        try:
            defense_type_multiplier = evalFunctions.evaluate_matchup(active_pkm.type, opp_active_pkm.type,
                                                    list(map(lambda m: m.type, opp_active_pkm.moves)))
        except Exception as e:
            import traceback
            traceback.print_exc()
        
        if defense_type_multiplier <= 1.0: 
            return move_id
    
        # considera il cambio pokemon
        matchup: List[float] = []
        not_fainted = False

        try:
            for j in range(len(bench)):
                if bench[j].hp == 0.0:
                    matchup.append(0.0)
                else:
                    not_fainted = True
                    matchup.append(
                        evalFunctions.evaluate_matchup(bench[j].type, opp_active_pkm.type, list(map(lambda m: m.type, bench[j].moves))))
        
            best_switch_matchup = int(np.max(matchup))
            best_switch = np.argmax(matchup)
            current_matchup = evalFunctions.evaluate_matchup(active_pkm.type, opp_active_pkm.type,list(map(lambda m: m.type, active_pkm.moves)))
        except Exception as e: 
            import traceback
            traceback.print_exc()
            
        if not_fainted and best_switch_matchup >= current_matchup+1:
            return best_switch + 4
    
        return move_id

