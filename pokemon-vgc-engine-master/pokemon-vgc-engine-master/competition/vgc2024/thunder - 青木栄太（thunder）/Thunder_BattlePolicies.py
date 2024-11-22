from enum import IntEnum, auto
import random
from copy import deepcopy
from threading import Thread, Event

import numpy as np
from vgc.behaviour import BattlePolicy
from vgc.behaviour.BattlePolicies import BFSNode, estimate_damage, n_fainted, OneTurnLookahead
from vgc.competition.StandardPkmMoves import STANDARD_MOVE_ROSTER, Struggle
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, MAX_HIT_POINTS, TYPE_CHART_MULTIPLIER, DEFAULT_N_ACTIONS
from vgc.datatypes.Objects import GameState, Pkm, PkmMove, PkmRoster, PkmTeam, Weather
from vgc.datatypes.Types import PkmStat, PkmStatus, PkmType, WeatherCondition
from statistics import mean
import time

from vgc.engine import PkmBattleEnv


def estimate_damage(move: PkmMove, pkm_type: PkmType, opp_pkm_type: PkmType,
                    attack_stage: int, defense_stage: int, weather: WeatherCondition) -> float:
    move_type: PkmType = move.type
    move_power: float = move.power
    type_rate = TYPE_CHART_MULTIPLIER[move_type][opp_pkm_type]
    if type_rate == 0:
        return 0
    if move.fixed_damage > 0:
        return move.fixed_damage
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
    stage = (stage_level + 2.) / 2 if stage_level >= 0. else 2. / \
        (np.abs(stage_level) + 2.)
    damage = type_rate * \
        stab * weather * stage * move_power
    return damage


class ScoreEnum(IntEnum):
    RECOVER_MINUS = auto()
    MY_DEF_MINUS = auto()
    MY_ATK_MINUS = auto()
    OP_DEF_PLUS = auto()
    MY_SPD_MINUS = auto()
    OP_SPD_PLUS = auto()
    NONE = auto()
    OP_SPD_MINUS = auto()
    MY_SPD_PLUS = auto()
    OP_DEF_MINUS = auto()
    MY_ATK_PLUS = auto()
    MY_DEF_PLUS = auto()
    OP_ATK_MINUS = auto()
    RECOVER_PLUS = auto()
    SLEEP = auto()
    PARALYZED = auto()
    CONFUSED = auto()
    FROZEN = auto()
    BURNED = auto()
    POISONED = auto()


class MoveScore():
    def __init__(self, move_i, move: PkmMove, consider_speed: bool) -> None:
        self.move_i = move_i
        self.move = move
        self.target_bad = (move.stage < 0 or move.status != 0)
        self.target_good = (move.stage > 0)
        self.yuuri = "2draw"
        if (
            (move.target == 1 and self.target_bad) or
            (move.target == 0 and self.target_good) or
            move.recover > 0
        ):
            self.yuuri = "3my"
        elif (
            (move.target == 0 and self.target_bad) or
            (move.target == 1 and self.target_good) or
            move.recover < 0 or
            move.status != PkmStatus.NONE
        ):
            self.yuuri = "1opp"
        score_enum = ScoreEnum.NONE
        if move.recover > 0:
            score_enum = ScoreEnum.RECOVER_PLUS
        elif move.recover < 0:
            score_enum = ScoreEnum.RECOVER_MINUS
        elif move.target == 0:
            if move.stage < 0:
                if move.stat == PkmStat.ATTACK:
                    score_enum = ScoreEnum.MY_ATK_MINUS
                elif move.stat == PkmStat.DEFENSE:
                    score_enum = ScoreEnum.MY_DEF_MINUS
                elif move.stat == PkmStat.SPEED:
                    score_enum = ScoreEnum.MY_SPD_MINUS
            elif move.stage > 0:
                if move.stat == PkmStat.ATTACK:
                    score_enum = ScoreEnum.MY_ATK_PLUS
                elif move.stat == PkmStat.DEFENSE:
                    score_enum = ScoreEnum.MY_DEF_PLUS
                elif move.stat == PkmStat.SPEED:
                    score_enum = ScoreEnum.MY_SPD_PLUS
        elif move.target == 1:
            if move.stage < 0:
                if move.stat == PkmStat.ATTACK:
                    score_enum = ScoreEnum.OP_ATK_MINUS
                elif move.stat == PkmStat.DEFENSE:
                    score_enum = ScoreEnum.OP_DEF_MINUS
                elif move.stat == PkmStat.SPEED:
                    score_enum = ScoreEnum.OP_SPD_MINUS
            elif move.stage > 0:
                if move.stat == PkmStat.ATTACK:
                    score_enum = ScoreEnum.OP_ATK_PLUS
                elif move.stat == PkmStat.DEFENSE:
                    score_enum = ScoreEnum.OP_DEF_PLUS
                elif move.stat == PkmStat.SPEED:
                    score_enum = ScoreEnum.OP_SPD_PLUS
            elif move.status == PkmStatus.SLEEP:
                score_enum = ScoreEnum.SLEEP
            elif move.status == PkmStatus.PARALYZED:
                score_enum = ScoreEnum.PARALYZED
            elif move.status == PkmStatus.CONFUSED:
                score_enum = ScoreEnum.CONFUSED
            elif move.status == PkmStatus.FROZEN:
                score_enum = ScoreEnum.FROZEN
            elif move.status == PkmStatus.BURNED:
                score_enum = ScoreEnum.BURNED
            elif move.status == PkmStatus.POISONED:
                score_enum = ScoreEnum.POISONED

        score_order = [
            ScoreEnum.RECOVER_MINUS,
            ScoreEnum.NONE,
            # ScoreEnum.OP_SPD_MINUS,
            # ScoreEnum.MY_SPD_PLUS,
            # ScoreEnum.SLEEP,
            # ScoreEnum.POISONED,
            ScoreEnum.PARALYZED,
            ScoreEnum.RECOVER_PLUS,
            ScoreEnum.OP_DEF_MINUS,
        ]
        # if consider_speed:
        # else:

        if score_enum not in score_order:
            score_enum = ScoreEnum.NONE

        score = -1
        for order, senum in enumerate(score_order):
            if senum == score_enum:
                score = order
                break
        self.score = score

        self.score_enum = score_enum


def find_best_first_move(enemy_hp,
                         damages: list[tuple[int, int, float, PkmMove]], consider_speed=False):
    """
    Determine the best move to use on the first turn to minimize the total number of turns needed to reduce the enemy's HP to zero.

    Parameters:
        enemy_hp (int): The initial HP of the enemy.

    Returns:
        dict: The best move to use on the first turn.
    """
    # Calculate minimum turns for each HP level using DP
    hp_step = 5
    dp = [float('inf')] * (enemy_hp + hp_step)
    dp[0] = 0  # No turns needed to reduce HP 0 to zero

    best_move_i = None

    best_move_is = []
    best_move_scores=[]
    for hp in range(hp_step, enemy_hp + hp_step, hp_step):
        for move_i, damage, acc, move in damages:
            # print("*"*40)
            # Calculate remaining HP after this move
            effective_hp = max(0, hp - damage)
            expected_extra_turns = 1/acc
            expected_turns = dp[effective_hp]+expected_extra_turns
            # success_turns = dp[effective_hp] + \
            #     1  # Additional turn if move hits
            # expected_turns = success_turns / acc  # Adjust for accuracy
            if expected_turns < dp[hp]:
                dp[hp] = expected_turns  # Update dp table
                best_move_is = []
            if hp == enemy_hp:
                if expected_turns <= dp[hp]:
                    best_move_is.append((move_i, move))

            # print(f"{damage},{effective_hp},{expected_extra_turns},{expected_turns}")
            # print(f"dp[{hp}] <- {dp[hp]} (updated)")

            # print(f"dp[hp] to be {dp[hp]}")
        # print(hp, dp[hp])
    # assert (best_move_i is not None)
    if len(best_move_is) > 0:
        best_move_scores = [MoveScore(best_move_i, best_move, consider_speed=consider_speed)
                            for best_move_i, best_move in best_move_is]
        best_move_scores = sorted(
            best_move_scores, key=lambda x: (x.yuuri, x.score), reverse=True)
        # best_move_scores, key=lambda x: (x.yuuri, x.score_enum), reverse=True)
        best_score = best_move_scores[0]
        best_move_i, best_move = best_score.move_i, best_score.move
        if len(best_move_scores) >= 2:
            pass
        # best_move_i, best_move = best_move_is[0]

    for hp in range(hp_step, enemy_hp + hp_step):
        if dp[hp] == 0:
            dp[hp] = 100
    assert (dp[enemy_hp] != 0)
    return best_move_i, dp[enemy_hp],best_move_scores


def get_max_damage(
    weather: Weather,
    my_pkm: Pkm, my_stage: list[PkmStat],
    op_pkm: Pkm, op_stage: list[PkmStat]
):
    damages = []
    for move_i, move in enumerate(my_pkm.moves):
        if move.pp == 0:
            move = Struggle
        if move is Struggle or move.name is None:
            continue
        damage = estimate_damage(move, my_pkm.type, op_pkm.type,
                                 my_stage[PkmStat.ATTACK], op_stage[PkmStat.DEFENSE], weather)
        damages.append(damage)

    return max(damages+[0])


def get_dp_move_turn(
    weather: Weather,
    my_pkm: Pkm, my_stage: list[PkmStat],
    op_pkm: Pkm, op_stage: list[PkmStat]
):
    if my_pkm.hp == 0 or op_pkm.hp == 0:
        return None, None,None
    damages = []
    for move_i, move in enumerate(my_pkm.moves):
        if move.pp == 0:
            move = Struggle
        if move is Struggle or move.name is None:
            continue
        # print(move, move.power)
        damage = estimate_damage(move, my_pkm.type, op_pkm.type,
                                 my_stage[PkmStat.ATTACK], op_stage[PkmStat.DEFENSE], weather)
        damages.append((move_i, int(damage), move.acc, move))
    if len(damages) == 0:
        return None, None,None
    # print(damages)
    move_i, best_turn,best_move_scores = find_best_first_move(
        int(op_pkm.hp), damages, my_stage[PkmStat.SPEED] < op_stage[PkmStat.SPEED])

    return move_i, best_turn,best_move_scores



def n_fainted(t: PkmTeam):
    fainted = 0
    fainted += t.active.hp == 0
    if len(t.party) > 0:
        fainted += t.party[0].hp == 0
    if len(t.party) > 1:
        fainted += t.party[1].hp == 0
    return fainted


class ThunderPlayer(BattlePolicy):
    """
    """

    def __init__(self, roster: PkmRoster = None):
        self.roster = roster
        self.max_elapesed = -1

    def set_estimate_op(self, op: Pkm):

        pkms = [pkm for pkm in self.roster
                if (op.max_hp == pkm.max_hp and op.type == pkm.type)]
        # assert (len(pkms) != 0)
        if len(pkms) != 0:
            op.moves = pkms[0].moves[:]

    def get_action(self, g: PkmBattleEnv.PkmBattleEnv) -> int:  # g: PkmBattleEnv

        my_team = g.teams[0]
        op_team = g.teams[1]
        my_active = my_team.active
        op_active = op_team.active

        if self.roster is not None:
            self.set_estimate_op(op_active)
        else:
            for move_i in range(4):
                if op_active.moves[move_i].name is None:
                    if move_i==0:
                        type_moves=[move for move in STANDARD_MOVE_ROSTER if move.type==op_active.type]
                        op_active.moves[move_i]=random.choice(type_moves)
                    else:
                        op_active.moves[move_i]=random.choice(STANDARD_MOVE_ROSTER)
        for i, move in enumerate(my_active.moves):
            if not move.priority:
                continue
            damage = estimate_damage(move, my_active.type, op_active.type,
                                     g.teams[0].stage[PkmStat.ATTACK],
                                     g.teams[1].stage[PkmStat.DEFENSE],
                                     g.weather.condition)
            if damage >= op_active.hp:
                return i

        if self.roster is not None:
            my_move_i, my_turn,best_move_scores = get_dp_move_turn(
                g.weather,
                my_team.active, my_team.stage,
                op_active, op_team.stage)
            op_move_i, op_turn,best_move_scores = get_dp_move_turn(
                g.weather,
                op_team.active, op_team.stage,
                my_team.active, my_team.stage)

            for move_i, move in enumerate(my_active.moves):
                if move.recover > 0:
                    recover_max = max(
                        move.recover, my_active.max_hp-my_active.hp)
                    op_max_damage = get_max_damage(
                        g.weather,
                        op_team.active, op_team.stage,
                        my_team.active, my_team.stage)
                    if recover_max > op_max_damage:
                        return move_i

            if my_move_i is not None and (op_turn is None or my_turn < op_turn):
                return my_move_i

        my_pkm=my_team.active

        my_move_i, my_best_turn,my_best_move_scores = get_dp_move_turn(
            g.weather,
            my_pkm, my_team.stage,
            op_team.active, op_team.stage)

        op_move_i, op_best_turn,op_best_move_scores = get_dp_move_turn(
            g.weather,
            op_team.active, op_team.stage,
            my_pkm, my_team.stage)
        
        if my_best_turn is not None and my_best_turn >= 3:
            for i, move in enumerate(my_active.moves):
                if move.status != PkmStatus.SLEEP:
                    continue
                if op_active.status == PkmStatus.SLEEP:
                    continue
                if TYPE_CHART_MULTIPLIER[move.type][op_active.type] <= 0:
                    continue
                return i

        if my_move_i is None:
            for pkm_i, pkm in enumerate(my_team.party):
                if not pkm.fainted():
                    return 4+pkm_i
            return 0
        return my_move_i
