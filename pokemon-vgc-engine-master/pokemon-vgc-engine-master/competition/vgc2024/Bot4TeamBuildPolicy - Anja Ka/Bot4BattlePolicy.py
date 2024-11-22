from typing import List
import numpy as np

from vgc.competition.StandardPkmMoves import Struggle
from vgc.behaviour import BattlePolicy, BattlePolicies
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER, DEFAULT_N_ACTIONS, MAX_HIT_POINTS, STATE_DAMAGE, SPIKES_2, SPIKES_3
from vgc.datatypes.Objects import GameState, PkmTeam
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition, PkmStatus, PkmEntryHazard, MAX_STAGE, MIN_STAGE
from vgc.datatypes.Objects import PkmMove, Pkm
from operator import itemgetter

class Bot4BattlePolicy(BattlePolicy):

    def __init__(self, switch_probability: float = .15, n_moves: int = DEFAULT_PKM_N_MOVES,
                 n_switches: int = DEFAULT_PARTY_SIZE):
        super().__init__()
        

    def get_action(self, g: GameState):
        # get weather condition
        weather = g.weather.condition

        # get my pokémon
        my_team = g.teams[0]
        my_active = my_team.active
        my_attack_stage = my_team.stage[PkmStat.ATTACK]
        my_defense_stage = my_team.stage[PkmStat.DEFENSE]

        # get opp team
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_active_type = opp_active.type
        opp_attack_stage = opp_team.stage[PkmStat.ATTACK]
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]


        #priority/speed win
        attack_order = self.CanAttackFirst(my_team, opp_team, my_active, opp_active)
        if attack_order == 0:
            for move in my_active.moves:
                if move.priority and self.calculate_damage(move, my_active.type, opp_active_type, my_attack_stage, opp_defense_stage, weather) >= opp_active.hp:
                    return my_active.moves.index(move)
        
        # get most damaging move from my active pokémon
        damage: List[float] = []
        for move in my_active.moves:
            try:
                damage.append(self.calculate_damage(move, my_active.type, opp_active_type,
                                          my_attack_stage, opp_defense_stage, weather))
            except:
                #print("Something is wrong")
                damage.append(-1)

        max_move_id = int(np.argmax(damage))


        #if victory is immediately possible, defeat opponent
        if damage[max_move_id] >= opp_active.hp and attack_order >=0:
            damage_sorted = sorted(damage)
            for dmg in damage_sorted:
                if dmg >= opp_active.hp:
                    move = damage.index(dmg)
                    if my_active.moves[move].acc >= my_active.moves[max_move_id].acc and my_active.moves[move].acc >= 0.7:
                        return move

        
        #calculate survival
        survivable_turns = self.estimate_survivable_turns(my_active, opp_active, my_defense_stage, opp_attack_stage, weather) #-10 means unknown
        #calculate turns till win
        turns_to_win = self.estimate_turns_till_win(my_active, opp_active, my_attack_stage, opp_defense_stage, weather)

        #control speed if possible
        if attack_order < 1:
            for move in my_active.moves:
                if move.stat == PkmStat.SPEED and move.target == 1 and move.stage < 0 and move.prob >= 0.8:
                    if TYPE_CHART_MULTIPLIER[move.type][opp_active.type] == 0:
                        print('Stop')
                    return my_active.moves.index(move)

         #try preventing the opp from attacking
        if not(opp_active.status in [PkmStatus.SLEEP, PkmStatus.FROZEN, PkmStatus.PARALYZED, PkmStatus.CONFUSED]  or opp_team.confused):
            for move in my_active.moves:
                if move.target == 1 and move.pp > 0 and move.acc >= 0.8:
                    if (move.status == PkmStatus.FROZEN and move.prob >= 0.5) or (move.status == PkmStatus.SLEEP and move.prob >= 0.8):
                        if self.check_status_application(move.status, opp_team):
                            return my_active.moves.index(move)
                    elif (move.status == PkmStatus.PARALYZED or move.status == PkmStatus.CONFUSED) and move.prob >= 0.8:
                        if self.check_status_application(move.status, opp_team):
                            return my_active.moves.index(move)                       
                      
        #boost DEF
        def_turn_boost = []
        if my_defense_stage < 4:
            for move in my_active.moves:
                if move.stat == PkmStat.DEFENSE and move.target == 0 and move.stage > 0 and move.prob >= 0.6:
                    def_turn_boost.append(self.estimate_survivable_turns(my_active, opp_active, my_defense_stage + move.stage, opp_attack_stage, weather) - survivable_turns + (-1 if attack_order < 1 else 0))
                else:
                    def_turn_boost.append(0)

        def_boost = def_turn_boost[int(np.argmax(def_turn_boost))]      
                    
        
        
        #debuff opponent
        # turn_boost = []
        # if opp_attack_stage > MIN_STAGE:
        #     if move.stat == PkmStat.ATTACK and move.target == 1 and move.stage < 0:
        #         turn_boost.append(self.estimate_survivable_turns(my_active, opp_active, my_defense_stage, opp_attack_stage  + move.stage, weather) - survivable_turns + (-1 if attack_order < 1 else 0))
        #     else:
        #         turn_boost.append(0)

        # if turn_boost[int(np.argmax(turn_boost))] > 0:
        #     return int(np.argmax(turn_boost))

        # turn_boost = []
        # if opp_defense_stage > MIN_STAGE:
        #     if move.stat == PkmStat.DEFENSE and move.target == 1 and move.stage < 0:
        #         turn_boost.append(turns_to_win - self.estimate_turns_till_win(my_active, opp_active, my_attack_stage, opp_defense_stage + move.stage, weather) - 1)              
        #     else:
        #         turn_boost.append(0)

        # if turn_boost[int(np.argmax(turn_boost))] > 0:
        #     return int(np.argmax(turn_boost))
                    

        #can't defeat opp, try switch
        if survivable_turns > -10:
            team_chance = []
            if (turns_to_win > survivable_turns and not self.might_not_attack(opp_active)) or my_team.confused:
                for p in my_team.party:
                    team_survive = self.estimate_survivable_turns(p, opp_active, 0, opp_attack_stage, weather)
                    team_win =  self.estimate_turns_till_win(p, opp_active, 0, opp_defense_stage, weather)
                    team_chance.append(team_win)
                    team_chance.append(team_survive)

                team_pkm_1 = -1
                team_pkm_2 = -1
                if team_chance[1] > 1: 
                    team_pkm_1 = team_chance[1] -team_chance[0]
                    if team_chance[3] > 1:
                        team_pkm_2 =  team_chance[3] - team_chance[2]
                if team_pkm_1 > 0 and team_pkm_1 >= team_pkm_2 and team_pkm_1 > (turns_to_win-survivable_turns):
                    if def_boost < team_pkm_1:
                        return 4
                    elif def_boost > 0:
                        return int(np.argmax(def_turn_boost))
                if team_pkm_2 > 0 and team_pkm_1 < team_pkm_2 and team_pkm_2 > (turns_to_win-survivable_turns):
                    if def_boost < team_pkm_2:
                        return 5
                    else:
                        return int(np.argmax(def_turn_boost))

        #boost atk if it pays off
        if def_boost > 0:
            if def_boost == 1 and my_defense_stage > 2 and damage[max_move_id]/opp_active.hp >= 0.20:
                return max_move_id
            return int(np.argmax(def_turn_boost))
        atk_turn_boost = []
        if my_attack_stage < 4 and survivable_turns > -10:
            for move in my_active.moves:
                if move.stat == PkmStat.ATTACK and move.target == 0 and move.stage > 0 and move.prob >= 0.6:
                    atk_turn_boost.append(turns_to_win - self.estimate_turns_till_win(my_active, opp_active, my_attack_stage + move.stage, opp_defense_stage, weather) - 1)
                else:
                    atk_turn_boost.append(0)

            if atk_turn_boost[int(np.argmax(atk_turn_boost))] > 0:
                return int(np.argmax(atk_turn_boost))

        return max_move_id

    def estimate_survivable_turns(self, pkm:Pkm, opp:Pkm, own_def_stage:int, opp_atk_stage:int, weather):
        turns:int = 0
        hp:float = pkm.hp
        
        [move_ids, opp_dmg] = self.get_max_damage_moves_sorted(opp, pkm, opp_atk_stage, own_def_stage, weather)
        max_dmg_move = move_ids[0] 
        max_dmg_move_index = 0
        pp_cost = [0, 0, 0, 0]
        if opp.moves[max_dmg_move].name == None:
            return -10
        while hp > 0 and turns < 20:
            turns += 1
            if opp.moves[max_dmg_move].pp - pp_cost[max_dmg_move] > 0:
                hp -= opp_dmg[max_dmg_move]
                pp_cost[max_dmg_move] += 1
            else:
                max_dmg_move_index += 1
                max_dmg_move = move_ids[max_dmg_move]
                hp -= opp_dmg[max_dmg_move]
                pp_cost[max_dmg_move] += 1
            if (opp.moves[max_dmg_move].stat == PkmStat.DEFENSE and opp.moves[max_dmg_move].target == 1) or (opp.moves[max_dmg_move].stat == PkmStat.ATTACK and opp.moves[max_dmg_move].target == 0):
                [move_ids, opp_dmg] = self.get_max_damage_moves_sorted(opp, pkm, opp_atk_stage, own_def_stage, weather)

        return turns if turns < 20 else -10
    
    def estimate_turns_till_win(self, pkm:Pkm, opp:Pkm, own_atk_stage:int, opp_def_stage:int, weather) -> int:
        return self.estimate_survivable_turns(opp, pkm, opp_def_stage, own_atk_stage, weather)
            
    def might_not_attack(self, pkm:Pkm):
        if pkm.status in [PkmStatus.CONFUSED, PkmStatus.FROZEN, PkmStatus.PARALYZED, PkmStatus.SLEEP]:
            return True
        else:
            return False
    
    def calculate_damage(self, move: PkmMove, pkm_type: PkmType, opp_pkm_type: PkmType, attack_stage: int, defense_stage: int, weather: WeatherCondition) -> float:
        if move.pp <= 0:
            move = Struggle
        
        fixed_damage = move.fixed_damage
        if fixed_damage > 0. and TYPE_CHART_MULTIPLIER[move.type][opp_pkm_type] > 0.:
            damage = fixed_damage
        else:
            stab = 1.5 if move.type == pkm_type else 1.
            if (move.type == PkmType.WATER and weather == WeatherCondition.RAIN) or (move.type == PkmType.FIRE and weather == WeatherCondition.SUNNY):
                weather = 1.5
            elif (move.type == PkmType.WATER and weather == WeatherCondition.SUNNY) or (move.type == PkmType.FIRE and weather == WeatherCondition.RAIN):
                weather = .5
            else:
                weather = 1.       
        
            stage_level = attack_stage - defense_stage
            stage = (stage_level + 2.) / 2 if stage_level >= 0. else 2. / (np.abs(stage_level) + 2.)
            multiplier = TYPE_CHART_MULTIPLIER[move.type][opp_pkm_type] if move != Struggle else 1.0
            damage = multiplier * stab * weather * stage * move.power
        return round(damage)
    
    #-1 lower speed, 0 same speed, or enemy has prio, 1 higher speed and opponent has no prio
    def CanAttackFirst(self, my_team:PkmTeam, opp_team:PkmTeam, my_active:Pkm, opp_active:Pkm) -> int:
        """
        Get attack order for this turn.

        :return: -2 opp faster and has priority, -1 opp faster, 1 self faster and no opp prio, 0 same speed, 0.5 if faster but opp prio
        """
        speed0 = my_team.stage[PkmStat.SPEED]
        speed1 = opp_team.stage[PkmStat.SPEED]

        opp_might_act_earlier = False
        for opp_poss_act in opp_active.moves:
            if opp_poss_act.priority:
                opp_might_act_earlier = True

        if speed1 > speed0:
            if opp_might_act_earlier:
                return -2
            return -1
        if speed0 > speed1 and not opp_might_act_earlier:
            return 1
        if speed0 > speed1 and opp_might_act_earlier:
            return 0.5
        else:
            return 0

    def get_switch_opp_greedy(self, my_team:PkmTeam, opp_active:Pkm, opp_move_id: int, opp_attack_stage:int, weather):       
        #evaluate wich of my pokemon would get hurt less
        best_state = -10
        index = 0
        for pkm in my_team.party:
            if pkm != my_team.active and pkm.hp > 0:
                state = (pkm.hp - self.calculate_damage(opp_active.moves[opp_move_id],  opp_active.type, pkm.type, opp_attack_stage, 0, weather)) / pkm.max_hp
                if state > best_state:
                    index = my_team.party.index(pkm)
                    best_state = state
        
        if best_state > 0 and index > 0:
            return index
        else: 
            return 0

    def get_possible_damages(self, attacker: Pkm, defender: Pkm, attack_stage: int, defense_stage: int, weather) -> list[float]:
        damage: List[float] = []
        for move in attacker.moves:
            try:
                damage.append(self.calculate_damage(move, attacker.type, defender.type, attack_stage, defense_stage, weather))
                
            except:               
                damage.append(-1)
                pass
        return damage

    def get_max_damage_move(self, attacker: Pkm, defender: Pkm, attack_stage, defense_stage, weather) -> list[int, float]:
        damage = self.get_possible_damages(attacker, defender, attack_stage, defense_stage, weather)

        move_id = int(np.argmax(damage))

        return [move_id, damage[move_id]]
    
    def get_max_damage(self, attacker: Pkm, defender: Pkm, attack_stage, defense_stage, weather) -> int:
        damage = self.get_possible_damages(attacker, defender, attack_stage, defense_stage, weather)

        move_id = int(np.argmax(damage))

        return damage[move_id]

    def get_max_damage_moves_sorted(self, attacker: Pkm, defender: Pkm, attack_stage, defense_stage, weather) -> list[list[int], list[float]]:
        damage = self.get_possible_damages(attacker, defender, attack_stage, defense_stage, weather)

        damage_set = [[i, damage[i]] for i in range(0, 4)]
        damage_sorted = sorted(damage_set, key=itemgetter(1), reverse=True)
        move_ids = [damage_sorted[0][0], damage_sorted[1][0], damage_sorted[2][0], damage_sorted[3][0]]

        return [move_ids, damage]

    def check_status_application(self, status: PkmStatus, opp_team: PkmTeam) -> bool:
        pkm = opp_team.active

        if status == PkmStatus.PARALYZED and pkm.type != PkmType.ELECTRIC and pkm.type != PkmType.GROUND and pkm.status != PkmStatus.PARALYZED:
            #print("Opponent can be paralyzed!")
            #pkm.status = PkmStatus.PARALYZED
            return True
        elif status == PkmStatus.POISONED and pkm.type != PkmType.POISON and pkm.type != PkmType.STEEL and pkm.status != PkmStatus.POISONED:
            #print("Opponent can be poisoned!")
            return True
        elif status == PkmStatus.BURNED and pkm.type != PkmType.FIRE and pkm.status != PkmStatus.BURNED:
            #print("Opponent can be burned!")
            return True
        elif status == PkmStatus.SLEEP and pkm.status != PkmStatus.SLEEP:
            #print("Opponent can be put asleep!")
            return True
        elif status == PkmStatus.FROZEN and pkm.type != PkmType.ICE and pkm.status != PkmStatus.FROZEN:
            #print("Opponent can be frozen solid!")
            return True
        elif not opp_team.confused:
            #print("Opponent can be confused!")
            return True
        
        return False
    
    def CheckSwitchWorstCase(self, my_team:PkmTeam, my_pkm: Pkm, opp_pkm: Pkm, opp_attack_stage: int, weather: WeatherCondition) -> float:
        damage = 0
        for move in opp_pkm.moves:
            move_damage = self.calculate_damage(move, opp_pkm.type, my_pkm.type, opp_attack_stage, 0 , weather)
            if move_damage > damage:
                damage = move_damage

        #look out for entry hazards and weather damage
        spikes = my_team.entry_hazard[PkmEntryHazard.SPIKES]
        if spikes > 0:
            damage += STATE_DAMAGE if spikes <= 1 else SPIKES_2 if spikes == 2 else SPIKES_3
        if weather == WeatherCondition.SANDSTORM and (my_pkm.type != PkmType.ROCK and my_pkm.type != PkmType.GROUND and my_pkm.type != PkmType.STEEL):
            damage += STATE_DAMAGE
        elif self.weather.condition == WeatherCondition.HAIL and (my_pkm.type != PkmType.ICE):
            damage += STATE_DAMAGE

        return damage