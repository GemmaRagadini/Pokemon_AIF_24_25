from typing import List
import numpy as np
from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER, DEFAULT_N_ACTIONS
from vgc.competition.StandardPkmMoves import STANDARD_MOVE_ROSTER
from vgc.datatypes.Objects import GameState, PkmMove, PkmEntryHazard, PkmTeam, PkmStatus, Pkm
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition


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
    

class BFSNode:

    def __init__(self):
        self.a = None
        self.g = None
        self.parent = None
        self.depth = 0
        self.eval = 0.0
    
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
   
class TeraBattlePolicy(BattlePolicy):
    def __init__(self, switch_probability: float = .15, n_moves: int = DEFAULT_PKM_N_MOVES,
                 n_switches: int = DEFAULT_PARTY_SIZE):
        super().__init__()
        self.hail_used = False
        self.sandstorm_used = False

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def calculate_damage(self, my_active, opp_active, my_attack_stage, opp_defense_stage, weather) -> List[float]:
        damage = []
        for move in my_active.moves:
            damage.append(estimate_damage(move.type, my_active.type, move.power, opp_active.type, my_attack_stage,
                                          opp_defense_stage, weather))
        return damage


    def evaluate_weather_move(self, weather, weather_condition, used_flag, immune_types, my_active, opp_active, my_party, opp_party, my_attack_stage, opp_defense_stage, opp_attack_stage, my_defense_stage, damage):
        if weather != weather_condition and not getattr(self, used_flag):
            weather_move = -1
            for i in range(DEFAULT_PKM_N_MOVES):
                if my_active.moves[i].weather == weather_condition:
                    weather_move = i
            immune_pkms = 0
            for pkm in my_party:
                if not pkm.fainted() and pkm.type in immune_types:
                    immune_pkms += 1
                    if damage[i] + estimate_damage(my_active.moves[i].type, my_active.type, my_active.moves[i].power, opp_active.type, my_attack_stage, opp_defense_stage, weather) >= opp_active.hp:
                        immune_pkms += 3  # Adding extra weight if current move can OHKO the opponent
            for move in opp_active.moves:
                if estimate_damage(move.type, opp_active.type, move.power, my_active.type, opp_attack_stage, my_defense_stage, weather) * 2 > my_active.hp:
                    immune_pkms += 3  # Adding extra weight if my active can survive two moves from opponent
            if weather_move != -1 and immune_pkms > 2:
                setattr(self, used_flag, True)
                return weather_move
        return None

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
        opp_party = opp_team.party
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

        # Evaluate type matchups
        best_damage_overall = -float('inf')
        best_move_id = move_id = int(np.argmax(damage))
        can_ohko_active = damage[move_id] >= opp_active.hp

        for i, move in enumerate(my_active.moves):
            active_type_multiplier = TYPE_CHART_MULTIPLIER[my_active.type][opp_active.type]
            total_damage = damage[i]
            if ((move.max_pp - move.pp) > 0 and active_type_multiplier >= 2.0) or move.type == my_active.type:
                for pkm in opp_team.party:
                    if not pkm.fainted():
                        bench_type_multiplier = TYPE_CHART_MULTIPLIER[my_active.type][pkm.type]
                        if bench_type_multiplier <= 0.5:
                            if can_ohko_active and pkm.hp >= estimate_damage(move.type, my_active.type, move.power, pkm.type, my_attack_stage,0, weather) + max(self.calculate_damage(my_active, pkm, my_attack_stage, 0, weather)):
                                best_damage_to_bench = -float('inf')
                                for j, move_bench in enumerate(my_active.moves):
                                    damage_to_bench = estimate_damage(move_bench.type, my_active.type, move_bench.power, pkm.type, my_attack_stage, opp_defense_stage, weather)
                                    if damage_to_bench > best_damage_to_bench and damage_to_bench < pkm.hp:
                                        best_damage_to_bench = damage_to_bench
                                        best_move_id = j
                        else:
                            total_damage += estimate_damage(move.type, my_active.type, move.power, pkm.type, my_attack_stage, opp_defense_stage, weather)
                if total_damage > best_damage_overall:
                    best_damage_overall = total_damage
                    best_move_id = i

        if best_damage_overall > damage[move_id]:
            move_id = best_move_id

        # If this damage is greater than the opponents current health we knock it out
        if damage[move_id] >= opp_active.hp:
            return move_id

        # If move is super effective use it
        if damage[move_id] > 0 and TYPE_CHART_MULTIPLIER[my_active.moves[move_id].type][opp_active.type] == 2.0:
            return move_id

        defense_type_multiplier = evaluate_matchup(my_active.type, opp_active.type,
                                                   list(map(lambda m: m.type, opp_active.moves)))

        if defense_type_multiplier <= 1.0:
            # Check for spike moves if spikes not set
            if opp_team.entry_hazard != PkmEntryHazard.SPIKES and opp_not_fainted_pkms > DEFAULT_PARTY_SIZE / 2:
                for i in range(DEFAULT_PKM_N_MOVES):
                    if my_active.moves[i].hazard == PkmEntryHazard.SPIKES:
                        return i

            # Use sandstorm if not set and you have pokemons immune to that
            sandstorm_move = self.evaluate_weather_move(weather, WeatherCondition.SANDSTORM, 'sandstorm_used', [PkmType.GROUND, PkmType.STEEL, PkmType.ROCK], my_active, opp_active, my_party, opp_party, my_attack_stage, opp_defense_stage, opp_attack_stage, my_defense_stage, damage)
            if sandstorm_move is not None:
                return sandstorm_move

            # Use hail if not set and you have pokemons immune to that
            hail_move = self.evaluate_weather_move(weather, WeatherCondition.HAIL, 'hail_used', [PkmType.ICE], my_active, opp_active, my_party, opp_party, my_attack_stage, opp_defense_stage, opp_attack_stage, my_defense_stage, damage)
            if hail_move is not None:
                return hail_move

            # Use hail if not set and you have pokemons immune to that
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
                    self.hail_used = True
                    return hail_move

            # If enemy attack and defense stage is 0, try to use attack or defense down
            if opp_attack_stage == 0 and opp_defense_stage == 0:
                for i in range(DEFAULT_PKM_N_MOVES):
                    if my_active.moves[i].target == 1 and my_active.moves[i].stage != 0 and (
                            my_active.moves[i].stat == PkmStat.ATTACK or my_active.moves[i].stat == PkmStat.DEFENSE):
                        return i

            return move_id

        # Find the best switch that will not fail
        best_switch = None
        best_switch_value = float('-inf')

        # Check each party member for a potential switch
        for p in range(DEFAULT_PARTY_SIZE):
            if not my_party[p].fainted():
                # Calculate type advantage
                switch_value = 0
                for move in opp_active.moves:
                    switch_value += TYPE_CHART_MULTIPLIER[move.type][my_party[p].type]
                if switch_value < best_switch_value:
                    best_switch_value = switch_value
                    best_switch = p

        # New logic to determine whether to switch or stay in battle
        if my_active.hp < 100 or my_active.hp <= max(self.calculate_damage(opp_active, my_active, opp_attack_stage, my_defense_stage, weather)):
            # If HP is less than 100, prefer to stay in and use the highest damage move
            can_ohko_bench = False
            for pkm in opp_team.party:
                if not pkm.fainted():
                    for move in my_active.moves:
                        if estimate_damage(move.type, my_active.type, move.power, pkm.type, my_attack_stage, opp_defense_stage, weather) >= pkm.hp:
                            can_ohko_bench = True
                            break
                if can_ohko_bench:
                    break
            if not can_ohko_bench:
                # Use the highest damage move if no bench Pokémon can be OHKO'd by the active Pokémon
                return move_id

        if best_switch is not None and my_party[best_switch] != my_active:
            return best_switch + 4

        return move_id