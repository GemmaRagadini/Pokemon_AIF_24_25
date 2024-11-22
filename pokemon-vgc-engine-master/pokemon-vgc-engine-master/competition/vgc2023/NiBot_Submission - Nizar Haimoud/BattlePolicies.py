import numpy as np

from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER
from vgc.datatypes.Objects import GameState
from vgc.datatypes.Types import PkmStat, PkmType


# Nizar's Bot
class NiBot(BattlePolicy):

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g: GameState):
        # Get weather condition
        weather = g.weather.condition

        # Get my Pokémon team
        my_team = g.teams[0]
        my_pkms = [my_team.active] + my_team.party

        # Get opponent's team
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_active_type = opp_active.type
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        # Initialize variables for the best move and its damage
        best_move_id = -1
        best_damage = -np.inf

        # Iterate over all my Pokémon and their moves to find the most damaging move
        for i, pkm in enumerate(my_pkms):
            if i == 0:
                my_attack_stage = my_team.stage[PkmStat.ATTACK]
            else:
                my_attack_stage = 0

            for j, move in enumerate(pkm.moves):
                if pkm.hp == 0.0:
                    continue

                # Estimate the damage of the move
                damage = estimate_damage(move.type, pkm.type, move.power, opp_active_type, my_attack_stage,
                                         opp_defense_stage, weather)

                # Check if the current move has higher damage than the previous best move
                if damage > best_damage:
                    best_move_id = j
                    best_damage = damage

        # Decide between using the best move, switching to the first party Pokémon, or switching to the second party Pokémon
        if best_move_id < 4:
            return best_move_id  # Use the current active Pokémon's best damaging move
        elif 4 <= best_move_id < 8:
            return 4  # Switch to the first party Pokémon
        else:
            return 5  # Switch to the second party Pokémon


def estimate_damage(move_type, pkm_type, move_power, opp_pkm_type, my_attack_stage, opp_defense_stage, weather):
    # Precompute the type chart multipliers for the move type and opponent's Pokémon type
    type_chart_multipliers = TYPE_CHART_MULTIPLIER[move_type][opp_pkm_type]

    # Calculate the damage using the precomputed multipliers and other factors
    if opp_defense_stage != 0.0:
        damage = move_power * type_chart_multipliers * my_attack_stage / opp_defense_stage * weather
    else:
        damage = 0.0

    # Consider the opponent's type and the weather condition
    damage *= evaluate_matchup(opp_pkm_type, pkm_type, move_type)

    return damage


def evaluate_matchup(pkm_type: PkmType, opp_pkm_type: PkmType, move_type: PkmType) -> float:
    # Determine the defensive matchup
    defensive_matchup = TYPE_CHART_MULTIPLIER[pkm_type][move_type]

    # Consider the opponent's type and the weather condition
    defensive_matchup *= TYPE_CHART_MULTIPLIER[opp_pkm_type][pkm_type]

    return defensive_matchup
