import random
from typing import List

import numpy as np

from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER
from vgc.datatypes.Objects import GameState
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition


class KidMeBot(BattlePolicy):
    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g: GameState):
        # get weather condition
        weather = g.weather.condition

        # get my pkms
        my_team = g.teams[0]
        my_pkms = [my_team.active] + my_team.party

        # get opp team
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_active_type = opp_active.type
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        # Check if the current Pokémon has a move that does over 100 damage
        current_pkm = my_pkms[0]
        for move_id, move in enumerate(current_pkm.moves):
            move_damage = estimate_damage(move.type, current_pkm.type, move.power, opp_active_type,
                                          my_team.stage[PkmStat.ATTACK], opp_defense_stage, weather)
            if move_damage > 100:
                return move_id  # Use the move if it does over 100 damage

        # If no move does over 100 damage, switch to the Pokémon that has the highest damage move
        highest_damage = 0
        highest_damage_move_id = 0
        for pkm_id, pkm in enumerate(my_pkms):
            for move_id, move in enumerate(pkm.moves):
                move_damage = estimate_damage(move.type, pkm.type, move.power, opp_active_type,
                                              my_team.stage[PkmStat.ATTACK], opp_defense_stage, weather)
                if move_damage > highest_damage:
                    highest_damage = move_damage
                    highest_damage_move_id = move_id

        # Switch to the Pokémon with the highest damage move
        return highest_damage_move_id + 4  # Switch to the corresponding party Pokémon


class DamageFocusedBot(BattlePolicy):
    """
    Greedy heuristic based competition designed to encapsulate a greedy strategy that prioritizes damage output.
    Source: http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf
    """

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g: GameState):
        # get weather condition
        weather = g.weather.condition

        # get my pkms
        my_team = g.teams[0]
        my_pkms = [my_team.active] + my_team.party

        # get opp team
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_active_type = opp_active.type
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        # get most damaging move from all my pkms
        damage: List[float] = []

        my_attack_stage = my_team.stage[PkmStat.ATTACK]
        pkm = my_pkms[0]
        my_attack_stage = 0
        for move in pkm.moves:
            if pkm.hp == 0.0:
                damage.append(0.0)
            else:
                damage.append(estimate_damage(move.type, pkm.type, move.power, opp_active_type, my_attack_stage,
                                              opp_defense_stage, weather))
        move_id = int(np.argmax(damage))

        return move_id  # use current active pkm best damaging move


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
    defensive_matchup = 0.0
    for mtype in moves_type + [opp_pkm_type]:
        defensive_matchup = min(TYPE_CHART_MULTIPLIER[mtype][pkm_type], defensive_matchup)
    return defensive_matchup


class BFSNode:

    def __init__(self):
        self.a = None
        self.g = None
        self.parent = None
        self.depth = 0
        self.eval = 0.0


def minimax_eval(s: GameState, depth):
    mine = s.teams[0].active
    opp = s.teams[1].active
    return mine.hp / mine.max_hp - 3 * opp.hp / opp.max_hp - 0.3 * depth


class QLearningAI:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((num_actions, num_actions))

    def requires_encode(self):
        return False

    def close(self):
        pass

    def get_action(self, g):
        state = self.convert_state(g)

        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(range(self.num_actions))
        else:
            action = np.argmax(self.q_table[state])

        return action

    def convert_state(self, g):
        # Convert the game state to a suitable representation for Q-learning
        # Implement your own logic to convert the state if needed
        return 0  # Placeholder, replace with your own implementation

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_next_q)
        self.q_table[state][action] = new_q
