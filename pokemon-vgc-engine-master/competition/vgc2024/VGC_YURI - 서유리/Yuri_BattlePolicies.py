from tkinter import CENTER, DISABLED, NORMAL
from types import CellType
from typing import List

import numpy as np
from customtkinter import CTk, CTkButton, CTkRadioButton, CTkLabel

from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER, DEFAULT_N_ACTIONS
from vgc.datatypes.Objects import GameState, PkmTeam
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition, PkmEntryHazard, PkmStatus
from vgc.competition.Competitor import Competitor



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


class BattleTrackBot(BattlePolicy):
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
        opp_damage: List[float] = []
        for move in opp_active.moves:
            opp_damage.append(move.power * TYPE_CHART_MULTIPLIER[move.type][my_active.type])
        
        pred_opp_id = int(np.argmax(opp_damage))

        # estimate damage pkm moves
        damage: List[float] = []
        for move in my_active.moves:
            damage.append(estimate_damage(move.type, my_active.type, move.power, opp_active.type, my_attack_stage,
                                          opp_defense_stage, weather))

        # get most damaging move
        move_id = int(np.argmax(damage))

        #  If this damage is greater than the opponents current health we knock it out
        if damage[move_id] >= opp_active.hp:
            print("try to knock it out")
            return move_id

        # If move is super effective use it
        if damage[move_id] > 0 and TYPE_CHART_MULTIPLIER[my_active.moves[move_id].type][opp_active.type] == 2.0:
            print("Attack with supereffective")
            return move_id

        if opp_active.moves[pred_opp_id].type != NORMAL and opp_active.moves[pred_opp_id].power != 30:
            pred_opp_move = opp_active.moves[pred_opp_id]
            defense_type_multiplier = TYPE_CHART_MULTIPLIER[pred_opp_move.type][my_active.type]
        else:
            defense_type_multiplier = evaluate_matchup(my_active.type, opp_active.type, list(map(lambda m: m.type, opp_active.moves)))
        
        if defense_type_multiplier <= 1.0:
            # Check for spike moves if spikes not setted
            if opp_team.entry_hazard != PkmEntryHazard.SPIKES and opp_not_fainted_pkms > DEFAULT_PARTY_SIZE / 2:
                for i in range(DEFAULT_PKM_N_MOVES):
                    if my_active.moves[i].hazard == PkmEntryHazard.SPIKES:
                        print("Setting Spikes")
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
                    print("Using Sandstorm")
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
                    print("Using Hail")
                    self.hail_used = True
                    return hail_move

            # If enemy attack and defense stage is 0 , try to use attack or defense down
            if opp_attack_stage == 0 and opp_defense_stage == 0:
                for i in range(DEFAULT_PKM_N_MOVES):
                    if my_active.moves[i].target == 1 and my_active.moves[i].stage != 0 and (
                            my_active.moves[i].stat == PkmStat.ATTACK or my_active.moves[i].stat == PkmStat.DEFENSE):
                        print("Debuffing enemy")
                        return i

            # If spikes not set try to switch
            print("Attacking enemy to lower his hp")
            return move_id

        best_party_move_id = -1
        best_party_damage = -np.inf

        # Iterate over all my Pokémon and their moves to find the most damaging move
        for i, pkm in enumerate(my_party):
            for j, move in enumerate(pkm.moves):
                if pkm.hp == 0.0:
                    continue

                # Estimate the damage of the move
                party_damage = estimate_damage(move.type, pkm.type, move.power, opp_active.type, my_attack_stage,
                                        opp_defense_stage, weather)

                # Check if the current move has higher damage than the previous best move
                if party_damage > best_party_damage:
                    best_party_move_id = j
                    best_party_damage = party_damage

        # Decide between using the best move, switching to the first party Pokémon, or switching to the second party Pokémon
        if best_party_damage >= opp_active.hp:
            print("party에서 즉시 처치 가능한 공격 존재!!!")
            if best_party_move_id >= 0 and best_party_move_id < 4:
                print(my_party[0].type, " 으로 교체!!!")
                return 4
            else:
                print(my_party[1].type, " 으로 교체!!!")
                return 5

        # If we are not switch, find pokemon with resistance 
        print("party에서도 즉시 처치 가능한 공격 없음... -> 저항력 높은 포켓몬 찾아보기!!!")
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
            print("Switching to someone else")
            return best_switch + 4

        # If our party has no non fainted pkm, lets give maximum possible damage with current active
        print("살아있는 포켓몬 없음 ㅠㅠ")
        return move_id




class ChampionshipTrackBot(BattlePolicy):
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
        # get weather condition
        weather = g.weather.condition

        # get my pkms
        my_team = g.teams[0]
        my_active = my_team.active
        my_party = my_team.party
        my_attack_stage = my_team.stage[PkmStat.ATTACK]
        my_defense_stage = my_team.stage[PkmStat.DEFENSE]
        my_speed_stage = my_team.stage[PkmStat.SPEED]

        # get opp team
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_not_fainted_pkms = len(opp_team.get_not_fainted())
        opp_attack_stage = opp_team.stage[PkmStat.ATTACK]
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]
        opp_speed_stage = opp_team.stage[PkmStat.SPEED]

        # estimate damage pkm moves
        opp_damage: List[float] = []
        for move in opp_active.moves:
            opp_damage.append(move.power * TYPE_CHART_MULTIPLIER[move.type][my_active.type])
        
        pred_opp_id = int(np.argmax(opp_damage))

        # estimate damage pkm moves
        damage: List[float] = []
        for move in my_active.moves:
            damage.append(estimate_damage(move.type, my_active.type, move.power, opp_active.type, my_attack_stage,
                                          opp_defense_stage, weather))

        # get most damaging move
        move_id = int(np.argmax(damage))


        if opp_active.moves[pred_opp_id].type != NORMAL and opp_active.moves[pred_opp_id].power != 30:
            pred_opp_move = opp_active.moves[pred_opp_id]
            print("상대 예상 공격 (After IF): ", pred_opp_move)
            defense_type_multiplier = TYPE_CHART_MULTIPLIER[pred_opp_move.type][my_active.type]
            # Evaluate if either move can KO the other's pokemon
            if opp_damage[pred_opp_id] >= my_active.hp or damage[move_id] >= opp_active.hp:
                if damage[move_id] >= opp_active.hp and opp_damage[pred_opp_id] >= my_active.hp:
                    print("Both moves can KO")
                    if my_active.moves[move_id].priority or my_speed_stage > opp_speed_stage:
                        print("My move is faster or has priority")
                        return move_id
                elif damage[move_id] >= opp_active.hp:
                    print("My move can KO opponent")
                    return move_id
                elif opp_damage[pred_opp_id] >= my_active.hp:
                    print("Opponent's move can KO me")
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
                    if not_fainted and my_party[best_switch] != my_active and my_party[best_switch].hp > opp_damage[pred_opp_id]:
                        print("Switching to someone else")
                        return best_switch + 4
            
        else:
            defense_type_multiplier = evaluate_matchup(my_active.type, opp_active.type, list(map(lambda m: m.type, opp_active.moves)))
        
        #  If this damage is greater than the opponents current health we knock it out
        if damage[move_id] >= opp_active.hp:
            return move_id

        # If move is super effective use it
        if damage[move_id] > 0 and TYPE_CHART_MULTIPLIER[my_active.moves[move_id].type][opp_active.type] == 2.0:
            print("Attack with supereffective")
            return move_id
        
          

        if defense_type_multiplier <= 1.0: # 상대에게 받는 피해 데미지가 1 이하
            # Check for spike moves if spikes not setted
            # opp_team.entry_hazard: 상대 팀이 배치한 진입 장애물의 상태 -> SPIKES가 설정되었는지 확인
            # opp_not_fainted_pkms: 상대 팀에서 아직 쓰러지지 않은 포켓몬의 수
            if opp_team.entry_hazard != PkmEntryHazard.SPIKES and opp_not_fainted_pkms > DEFAULT_PARTY_SIZE / 2:
                for i in range(DEFAULT_PKM_N_MOVES):
                    if my_active.moves[i].hazard == PkmEntryHazard.SPIKES:
                        print("Setting Spikes")
                        return i
            

            # Use sandstorm if not setted and you have pokemons immune to that
            if weather != WeatherCondition.SANDSTORM and not self.sandstorm_used and defense_type_multiplier < 1.0:
                sandstorm_move = -1 # -1: 모래폭풍 기술 사용 X
                for i in range(DEFAULT_PKM_N_MOVES):
                    if my_active.moves[i].weather == WeatherCondition.SANDSTORM:
                        sandstorm_move = i
                immune_pkms = 0
                for pkm in my_party:
                    if not pkm.fainted() and pkm.type in [PkmType.GROUND, PkmType.STEEL, PkmType.ROCK]:
                        immune_pkms += 1
                
                # 내 팀에 지면(GROUND), 강철(STEEL), 바위(ROCK) 유형의 포켓몬이 3마리 이상이고 모래폭풍 기술을 가진 포켓몬이 있다면 모래폭풍을 사용
                if sandstorm_move != -1 and immune_pkms > 2:
                    print("Using Sandstorm")
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
                    print("Using Hail")
                    self.hail_used = True
                    return hail_move

            # If enemy attack and defense stage is 0 , try to use attack or defense down
            # 상대방의 공격과 방어 스테이지가 모두 0이라면, 공격 또는 방어를 떨어뜨리는 기술을 사용
            if opp_attack_stage == 0 and opp_defense_stage == 0:
                for i in range(DEFAULT_PKM_N_MOVES):
                    if my_active.moves[i].target == 1 and my_active.moves[i].stage != 0 and (
                            my_active.moves[i].stat == PkmStat.ATTACK or my_active.moves[i].stat == PkmStat.DEFENSE):
                        print("Debuffing enemy")
                        return i
            
            for i in range(DEFAULT_PKM_N_MOVES):
                if my_active.moves[i].status != None and TYPE_CHART_MULTIPLIER[i][opp_active.type] > 0:
                    print("특수 기술 ", my_active.moves[i].status, " 사용!!!")
                    return i
            ''''
            # 상대 속도 스테이지 떨어트리기
            if opp_speed_stage == 0:
                for i in range(DEFAULT_PKM_N_MOVES):
                    if my_active.moves[i].stage == PkmStat.SPEED:
                        print("Debuffing speed")
                        return i
            '''
            # If spikes not set try to switch
            print("특별한 기술 없음... 그냥 데미지 큰 기술 사용!!!")
            return move_id

        best_party_move_id = -1
        best_party_damage = -np.inf

        # Iterate over all my Pokémon and their moves to find the most damaging move
        for i, pkm in enumerate(my_party):
            for j, move in enumerate(pkm.moves):
                if pkm.hp == 0.0:
                    continue

                # Estimate the damage of the move
                party_damage = estimate_damage(move.type, pkm.type, move.power, opp_active.type, my_attack_stage,
                                        opp_defense_stage, weather)

                # Check if the current move has higher damage than the previous best move
                if party_damage > best_party_damage:
                    best_party_move_id = j
                    best_party_damage = party_damage

        # Decide between using the best move, switching to the first party Pokémon, or switching to the second party Pokémon
        if best_party_damage >= opp_active.hp:
            print("party에서 즉시 처치 가능한 공격 존재!!!")
            if best_party_move_id >= 0 and best_party_move_id < 4:
                print(my_party[0].type, " 으로 교체!!!")
                return 4
            else:
                print(my_party[1].type, " 으로 교체!!!")
                return 5

        # If we are not switch, find pokemon with resistance 
        print("party에서도 즉시 처치 가능한 공격 없음... -> 저항력 높은 포켓몬 찾아보기!!!")
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
            print("Switching to someone else")
            return best_switch + 4

        # If our party has no non fainted pkm, lets give maximum possible damage with current active
        print("살아있는 포켓몬 없음 ㅠㅠ")
        return move_id