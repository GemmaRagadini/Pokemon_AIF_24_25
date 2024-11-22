from vgc.behaviour import BattlePolicy

from vgc.datatypes.Objects import GameState, Pkm, PkmTeam, PkmMove
from vgc.datatypes.Types import PkmType, PkmStat, PkmStatus, WeatherCondition
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER

from typing import List

import numpy as np

#function that estimates damage to the opponent pkm
def estimate_damage(power_move: float, move_type: PkmType, my_pkm_type: PkmType, opp_pkm_type: PkmType,
                    weather: WeatherCondition, attack_mypkm: float, defense_opppkm: float) -> float:
    
    # gets type_effectiveness
    type_effectiveness = TYPE_CHART_MULTIPLIER[move_type][opp_pkm_type]
    
    # stab wether my pkm type = mv type
    if (my_pkm_type == move_type):
        stab = 1.5
    else:
        stab = 1.0
        
    #weather effect accordig to move type
    if ((move_type == PkmType.WATER and weather == WeatherCondition.RAIN) or
        (move_type == PkmType.FIRE and weather == WeatherCondition.SUNNY)):
        weather_effect = 1.5
        power_move = power_move * 2.0
        
    elif ((move_type == PkmType.WATER and weather == WeatherCondition.SUNNY) or
          (move_type == PkmType.FIRE and weather == WeatherCondition.RAIN)):
        weather_effect = 0.5
        power_move = power_move / 2.0
        
    else:
        weather_effect = 1.0
        
    # calculate stage
    # see if attack > defense or reverse
    stage_level = attack_mypkm - defense_opppkm
    
    # impact attack would cause (none--> is 0, double--> is 2, little bit higher... etc)
    if stage_level >= 0:
        stage = (stage_level + 2.0) / 2.0
        
    # defense > attack --> estimate the proportion that has reduced stat
    else:
        stage = 2.0 / (np.abs(stage_level) + 2.0)
        
    damage = weather_effect * stab * power_move * type_effectiveness * stage
       
    return damage

def known_moves(p: Pkm) -> List[PkmMove]:
    return list(filter(lambda m: m.name, p.moves))


#inherits from BattlePolicy, abstract class
class RorrisBattlePolicy(BattlePolicy):
    
    def __init__(self):
        self.switch_count = 0 # count how many times it has switched
        
        ##### GENES #####
        self.gene1 = 0.165 # determines low hp of my pkm
        self.gene2 = 0.141 # determines low hp of opp pkm
        self.gene3 = 0.872 # determines high hp of opp pkm
        self.gene4 = 0.269 # determines the limit of pp to throw a movement
        self.gene5 = 0.37 # determines how many times it can switch
        self.gene6 = 0.415 # weight condition 1
        self.gene7 = 0.54 # weight condition 2
        self.gene8 = 0.242 # weight condition 3
        self.gene9 = 0.428 # weight condition 4
        self.gene10 = 0.324 # weight condition 5
        
    #Use current Game state
    def requires_encode(self) -> bool:
        return False
    
    def close(self):
        pass
        
    ##### CONDITIONS TO EVALUATE IN THE CURRENT MOMENT ####
    # CONDITION 1: my pkm has low hp
    def my_pkm_low_hp(self, my_pkm: Pkm) -> bool:
        return (my_pkm.hp < (my_pkm.max_hp * self.gene1))

    # CONDITION 2: opp pkm has low hp
    def opp_pkm_low_hp(self, opp_pkm: Pkm) -> bool:
        return(opp_pkm.hp < (opp_pkm.max_hp * self.gene2))

    # CONDITION 3: opp pkm is super effective to my active pkm
    def may_be_supereffective(self, opp_pkm: Pkm, my_pkm: Pkm) -> bool:
        return any(TYPE_CHART_MULTIPLIER[move.type][my_pkm.type] == 2 for move in known_moves(opp_pkm))

    # CONDITION 4: opp pkm has high hp
    def opp_pkm_high_hp(self, opp_pkm: Pkm) -> bool:
        return(opp_pkm.hp >= (opp_pkm.max_hp * self.gene3))

    # CONDITION 5: opp pkm has an status that makes him lose turns
    def opp_pkm_has_status(self, opp_pkm: Pkm, opp_team: PkmTeam) -> bool:
        return(opp_pkm.status == PkmStatus.SLEEP or 
            opp_team.confused or
            opp_pkm.status == PkmStatus.FROZEN)
        

    # evaluate which conditions are satisfied in the current moment, it would return a list of 0 or 1. 
    # 1 means condition is satisfied, 0 means it is not
    def evaluate_battle(self, my_pkm: Pkm, opp_pkm: Pkm, opp_team: PkmTeam) -> List[int]:
        
        conditions_satisfied = [int(self.my_pkm_low_hp(my_pkm)),
                                int(self.opp_pkm_low_hp(opp_pkm)),
                                int(self.may_be_supereffective(opp_pkm, my_pkm)),
                                int(self.opp_pkm_high_hp(opp_pkm)),
                                int(self.opp_pkm_has_status(opp_pkm, opp_team))]
        
        return conditions_satisfied

    #check if a movement can be used according to pp that has left
    def pp_of_a_movement(self, move: PkmMove) -> bool:
        return (move.pp > move.max_pp * self.gene4)

    
    def switch_to_best(self, t: PkmTeam, opp: Pkm, g: GameState) -> int:
            best_selection = None
            for i, pkm in enumerate([t.active] + t.party):
                if pkm.fainted():
                    continue
                if not self.may_be_supereffective(opp, pkm) and self.may_be_supereffective(pkm, opp):
                    return i
                if self.may_be_supereffective(pkm, opp):
                    best_selection = i
                elif best_selection is None and not self.may_be_supereffective(opp, pkm):
                    best_selection = i
            if best_selection is not None:
                return best_selection
            else:
                if t.active.status == PkmStatus.CONFUSED:
                    return 1
                else:
                    return 0
              
    def get_action(self, g:GameState):
        
        #get weather condition
        weather_condition = g.weather.condition
        
        #get my pokemon team
        my_pkm_team = g.teams[0] # team 0 is my team
        my_active_pkm= my_pkm_team.active # get my active pkm (just one)
        my_active_moves = my_active_pkm.moves # get moves
        attack_active_pkm = my_pkm_team.stage[PkmStat.ATTACK] # attack stats of active pkm
        speed_active_pkm = my_pkm_team.stage[PkmStat.SPEED] # speed stats of active pkm
        
        #get opponent's pkm team
        opp_pkm_team = g.teams[1] #team 1 is opp's team
        opp_active_pkm = opp_pkm_team.active # get opp's active pkm
        deffense_opp_active_pkm = opp_pkm_team.stage[PkmStat.DEFENSE] # defense stats opp active pkm
        speed_opp_pkm = opp_pkm_team.stage[PkmStat.SPEED] # speed stats opp active pkm
    
        #find which conditions are satisfied currently
        conditions_satisfied = self.evaluate_battle(my_active_pkm, opp_active_pkm, opp_pkm_team)
        
        #array of weights that would be evaluated in each turn, initialized to 0 to determine the most important action to return.
        weight = [0, 0, 0, 0, 0]
       
        # CONDITION 1: my pkm has low hp
        if conditions_satisfied[0] == 1:
            # if im faster try to knock the opponent
            if(speed_active_pkm > speed_opp_pkm):
                weight[0] += self.gene6
                
            # if not, switch to a resistant one
            else:
                if self.switch_count >= (int(self.gene5) * 10):
                    weight[0] -= 0.5 #penalizes so it would not switch ever
                    
                else:
                    weight[0] += self.gene6

        # CONDITION 2: opp pkm has low hp
        if conditions_satisfied[1] == 1:
            # try to knock opp
            weight[1] += self.gene7

        # CONDITION 3: opp pkm is super effective to my active pkm
        if conditions_satisfied[2] == 1:
            # if it's greater, try to launch a movement that lowers its attack
            if(speed_active_pkm > speed_opp_pkm):
                weight[2] += self.gene8
                
            # otherwise, switch to the most resistant
            else:
                if self.switch_count >= (int(self.gene5) * 10):
                    weight[2] -= 0.5 #penalizes
                    
                else:
                    weight[2] += self.gene8

        # CONDITION 4: opp pkm has high hp 
        #try to put some status on him
        if conditions_satisfied[3] == 1:
            weight[3] += self.gene9

        # CONDITION 5: opp pkm has an status that makes him lose turns
        # try to launch moves that increase my attack or lower his defense
        if conditions_satisfied[4] == 1:
            weight[4] += self.gene10
        
        #get the highest weight to determine the most important action to return
        higher_weight = np.argmax(weight)
        #print("higher weight", higher_weight)
        
        # for w in weight:
        #     print(w)
        
        # calculate the most damaging move
        damage_active_pkm = []
        for move in my_active_moves:
            damage_active_pkm.append(estimate_damage(move.power, move.type, my_active_pkm.type, opp_active_pkm.type,
                                                        weather_condition, attack_active_pkm, deffense_opp_active_pkm))
                    
        most_damaging_move = np.argmax(damage_active_pkm)
        
        #calculate the movement which has more pp
        pp_moves = []
        for move in my_active_moves:
            pp_moves.append(move.pp)
                    
        move_more_pp = np.argmax(pp_moves)
        
        
        # IF CONDITION 1 IS SATISFIED: my pkm has low hp
        if higher_weight == 0:
            
            # if im faster try to knock the opponent
            if(speed_active_pkm > speed_opp_pkm):
                if damage_active_pkm[most_damaging_move] >= opp_active_pkm.hp:
                    if(self.pp_of_a_movement(my_active_moves[most_damaging_move])):
                        return most_damaging_move
                
                # if i cannot knock him, try to recover
                else:
                    for i, move in enumerate(my_active_moves):
                        if move.recover > 0:
                            if(self.pp_of_a_movement(my_active_moves[i])):
                                #print("FINAL ACTION: MY PKM HAS LOW HP, ITS FASTER, TRY TO RECOVER")
                                return i
                        
            # if not, switch to a resistant one
            else:
                if(self.switch_count < (int(self.gene5) * 10)):
                    to_switch = self.switch_to_best(my_pkm_team, opp_active_pkm, g)
                    if to_switch != 0:
                        #print("FINAL ACTION: MY PKM HAS LOW HP, ITS SLOWER, TRY TO SWITCH")
                        self.switch_count += 1 #update counts of switches
                        return to_switch + 3
            
            
        # IF CONDITION 2 IS SATISFIED: opp pkm has low hp
        elif higher_weight == 1:
            # throw the most damaging move
            if(self.pp_of_a_movement(my_active_moves[most_damaging_move])):
                return most_damaging_move
        
        # IF CONDITION 3 IS SATISFIED: opp pkm is super effective to my active pkm
        elif higher_weight == 2:
            
            # if my speed is higher, try to lower its attack
            if(speed_active_pkm > speed_opp_pkm):
                for i, move in enumerate(my_active_moves):
                    if move.stat == PkmStat.ATTACK and move.stage < 0:
                        if(self.pp_of_a_movement(my_active_moves[i])):
                            return i
            
            # if not, switch to a resistant one     
            else:
                if(self.switch_count < (int(self.gene5) * 10)):
                    to_switch = self.switch_to_best(my_pkm_team, opp_active_pkm, g)
                    if to_switch != 0:
                        self.switch_count += 1 #update counts of switches
                        #print("TRY TO SWITCH")
                        return to_switch + 3
            
        # IF CONDITION 4 IS SATISFIED: opp pkm has high hp
        elif higher_weight == 3:
            
            # try to set status 
            if(opp_active_pkm.status == PkmStatus.NONE):
                #FIND THE FROZEN MOVEMENTS --> MORE PROBABILITY TO LOSE TURNS
                for i, move in enumerate(my_active_moves):
                    if((move.status == PkmStatus.FROZEN and opp_active_pkm.type != PkmType.ICE) and
                       move.target == 1):
                        if(self.pp_of_a_movement(my_active_moves[i])):
                            return i
                    
                #FIND THE SLEEP MOVEMENTS --> LESS PROBABILITY TO LOSE TURNS
                for i, move in enumerate(my_active_moves):
                    if ((move.status == PkmStatus.SLEEP) and 
                        move.target == 1):
                        if(self.pp_of_a_movement(my_active_moves[i])):
                            return i
                    
                #FIND THE PARALYZE MOVEMENTS --> IT WOULD LOSE JUST 1 TURN, BUT THROW IT ONLY IF MY SPEED IS HIGHER     
                for i, move in enumerate(my_active_moves):
                    if((move.status == PkmStatus.PARALYZED and opp_active_pkm.type != PkmType.ELECTRIC) and
                        move.target == 1):
                        if(self.pp_of_a_movement(my_active_moves[i])):
                            return i
                
                #FIND THE BURNED OR POISONED MOVEMENTS --> OPP HP WOULD BE REDUCED IN EACH TURN 
                for i, move in enumerate(my_active_moves):
                    if (((move.status == PkmStatus.BURNED) or (move.status == PkmStatus.POISONED)) and 
                        move.target == 1):
                        if(self.pp_of_a_movement(my_active_moves[i])):
                            return i
                    
                      
            if (not opp_pkm_team.confused and opp_active_pkm.status != PkmStatus.NONE):
                #FIND DE CONFUSE MOVEMENTS --> IT CAN THROW MOVEMENTS OR LOSE TURNS + DAMAGE TO HIMSELF
                for i, move in enumerate(my_active_moves):
                    if (((move.status == PkmStatus.CONFUSED and not opp_pkm_team.confused)) and 
                        move.target == 1):
                        if(self.pp_of_a_movement(my_active_moves[i])):
                            return i
                
        # IF CONDITION 5 IS SATISFIED: opp pkm has an status that makes him lose turns
        elif higher_weight == 4:
            
            #try to increase my attack stats
            for i, move in enumerate(my_active_moves):
                if ((move.stat == PkmStat.ATTACK) and move.stage > 0 and move.target == 0):
                    if(self.pp_of_a_movement(my_active_moves[i])):
                        return i
            
            #try to decrese opp defense stats
            for i, move in enumerate(my_active_moves):
                if ((move.stat == PkmStat.DEFENSE) and move.stage < 0):
                    if(self.pp_of_a_movement(my_active_moves[i])):
                        return i
                
        
        #if no condition is satisfied, throw the most damaging move
        if(self.pp_of_a_movement(my_active_moves[most_damaging_move])):
            return most_damaging_move
        
        # if i cant, throw the most damaging move because of its pp, just throw the move with more pp
        return move_more_pp
                        
