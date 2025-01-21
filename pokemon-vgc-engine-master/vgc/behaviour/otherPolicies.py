import math
import hashlib
import numpy as np
import random
from typing import List
from vgc.behaviour import evalFunctions
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition
from vgc.datatypes.Objects import Pkm, GameState
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER, DEFAULT_N_ACTIONS
from vgc.behaviour import BattlePolicy
from copy import deepcopy


"""
Other implemented policies 
"""

class MyMinimax(BattlePolicy):

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.name = "My Minimax"

    def minimax(self, g, depth, is_maximizing_player):
        """
        Classic minimax
        """

        if depth == 0:
            # Evaluate the current state
            try:
                evaluation = evalFunctions.game_state_eval(g,depth)
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            return evaluation , None

        if is_maximizing_player: # MAX
            max_eval = float('-inf')
            best_action = None
            try:
                for i in range(DEFAULT_N_ACTIONS):
                    g_copy = deepcopy(g)
                    s, _, _, _, _ = g_copy.step([i, 99])  # The opponent performs an invalid action that does not change the situation
                    if evalFunctions.n_defeated(s[0].teams[0]) > evalFunctions.n_defeated(g.teams[0]): 
                        continue # Ignore states where our number of defeated Pokémon increases
                    eval_score, _ = self.minimax(s[0], depth - 1, False)
                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_action = i
            except Exception as e:
                import traceback
                traceback.print_exc()

            return max_eval, best_action

        else:  # MIN
            min_eval = float('inf')
            best_action = None
            try:
                for j in range(DEFAULT_N_ACTIONS):
                    g_copy = deepcopy(g)
                    s, _, _, _, _ = g_copy.step([99, j])  
                    if evalFunctions.n_defeated(s[0].teams[1]) > evalFunctions.n_defeated(g.teams[1]):
                        continue
                    eval_score, _ = self.minimax(s[0], depth - 1, True)
                    if eval_score < min_eval:
                        min_eval = eval_score
                        best_action = j
            except Exception as e:
                import traceback
                traceback.print_exc()

            return min_eval, best_action


    def get_action(self, g) -> int:
        """
        Find the best action for MAX
        """
        try:
            _, best_action = self.minimax(g, self.max_depth, True)
        except Exception as e:
            import traceback
            traceback.print_exc()

        return best_action if best_action is not None else 0



class MyMinimaxWithAlphaBetaKiller(BattlePolicy):
    """
    Minimax algorithm with alpha beta pruning and killer moves optimization
    """
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.name = "Minimax with pruning alpha beta killer"
        self.killer_moves = {depth: [] for depth in range(max_depth + 1)}  

    def minimax(self, g, depth, alpha, beta, is_maximizing_player):
        if depth == 0:
            try:
                evaluation = evalFunctions.game_state_eval(g,depth)
            except Exception as e:
                import traceback
                traceback.print_exc() 
            return evaluation, None

        if is_maximizing_player:
            max_eval = float('-inf')
            best_action = None

            # Possible actions
            moves = list(range(DEFAULT_N_ACTIONS))
            # prioritizing killer moves
            killer_moves = self.killer_moves.get(depth, [])
            moves = sorted(moves, key=lambda move: move in killer_moves, reverse=True)

            try:
                for i in moves:
                    g_copy = deepcopy(g)
                    s, _, _, _, _ = g_copy.step([i, 99])
                    if evalFunctions.n_defeated(s[0].teams[0]) > evalFunctions.n_defeated(g.teams[0]):
                        continue

                    eval_score, _ = self.minimax(s[0], depth - 1, alpha, beta, False)
                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_action = i

                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        # update killer moves
                        if i not in self.killer_moves[depth]:
                            self.killer_moves[depth].append(i)
                            if len(self.killer_moves[depth]) > 2:  
                                self.killer_moves[depth].pop(0)
                        break
            except Exception as e:
                import traceback
                traceback.print_exc()

            return max_eval, best_action

        else:
            min_eval = float('inf')
            best_action = None

            moves = list(range(DEFAULT_N_ACTIONS))

            try:
                killer_moves = self.killer_moves.get(depth, [])
                moves = sorted(moves, key=lambda move: move in killer_moves, reverse=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            try:
                for j in moves:
                    g_copy = deepcopy(g)
                    s, _, _, _, _ = g_copy.step([99, j])
                    if evalFunctions.n_defeated(s[0].teams[1]) > evalFunctions.n_defeated(g.teams[1]):
                        continue

                    eval_score, _ = self.minimax(s[0], depth - 1, alpha, beta, True)
                    if eval_score < min_eval:
                        min_eval = eval_score
                        best_action = j

                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        if j not in self.killer_moves[depth]:
                            self.killer_moves[depth].append(j)
                            if len(self.killer_moves[depth]) > 2: 
                                self.killer_moves[depth].pop(0)
                        break
            except Exception as e:
                import traceback
                traceback.print_exc()

            return min_eval, best_action

    def get_action(self, g) -> int:
        try:
            _, best_action = self.minimax(g, self.max_depth, float('-inf'), float('inf'), True)
        except Exception as e:
                import traceback
                traceback.print_exc()

        return best_action if best_action is not None else 0



class MyMinimaxWithAlphaBetaKiller_my_eval(BattlePolicy):
    """
    Minimax algorithm with alpha beta pruning and killer moves optimization, using my_eval_fun
    """
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.name = "Minimax with pruning alpha beta killer and my eval"
        self.killer_moves = {depth: [] for depth in range(max_depth + 1)} 

    def minimax(self, g, depth, alpha, beta, is_maximizing_player):
        if depth == 0:
            try:
                evaluation = evalFunctions.my_eval_fun(g,depth)
            except Exception as e:
                import traceback
                traceback.print_exc() 
            return evaluation, None

        if is_maximizing_player:
            max_eval = float('-inf')
            best_action = None

            # possible actions
            moves = list(range(DEFAULT_N_ACTIONS))

            # prioritizing killer moves
            killer_moves = self.killer_moves.get(depth, [])
            moves = sorted(moves, key=lambda move: move in killer_moves, reverse=True)

            try:
                for i in moves:
                    g_copy = deepcopy(g)
                    s, _, _, _, _ = g_copy.step([i, 99])
                    if evalFunctions.n_defeated(s[0].teams[0]) > evalFunctions.n_defeated(g.teams[0]):
                        continue

                    eval_score, _ = self.minimax(s[0], depth - 1, alpha, beta, False)
                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_action = i

                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        # update killer moves
                        if i not in self.killer_moves[depth]:
                            self.killer_moves[depth].append(i)
                            if len(self.killer_moves[depth]) > 2:  
                                self.killer_moves[depth].pop(0)
                        break
            except Exception as e:
                import traceback
                traceback.print_exc()

            return max_eval, best_action

        else:
            min_eval = float('inf')
            best_action = None
            moves = list(range(DEFAULT_N_ACTIONS))

            try:
                killer_moves = self.killer_moves.get(depth, [])
                moves = sorted(moves, key=lambda move: move in killer_moves, reverse=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            try:
                for j in moves:
                    g_copy = deepcopy(g)
                    s, _, _, _, _ = g_copy.step([99, j])
                    if evalFunctions.n_defeated(s[0].teams[1]) > evalFunctions.n_defeated(g.teams[1]):
                        continue

                    eval_score, _ = self.minimax(s[0], depth - 1, alpha, beta, True)
                    if eval_score < min_eval:
                        min_eval = eval_score
                        best_action = j

                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        if j not in self.killer_moves[depth]:
                            self.killer_moves[depth].append(j)
                            if len(self.killer_moves[depth]) > 2: 
                                self.killer_moves[depth].pop(0)
                        break
            except Exception as e:
                import traceback
                traceback.print_exc()

            return min_eval, best_action

    def get_action(self, g) -> int:
        try:
            _, best_action = self.minimax(g, self.max_depth, float('-inf'), float('inf'), True)
        except Exception as e:
                import traceback
                traceback.print_exc()

        return best_action if best_action is not None else 0


class MyMinimax_my_eval(BattlePolicy):

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.name = "My Minimax with my eval"

    def minimax(self, g, depth, is_maximizing_player):
        """
        Classic minimax algorithm, using my_eval_fun
        """

        if depth == 0:
            # evaluate current status 
            try:
                evaluation = evalFunctions.my_eval_fun(g,depth)
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            return evaluation , None

        if is_maximizing_player: #MAX
            max_eval = float('-inf')
            best_action = None
            try:
                for i in range(DEFAULT_N_ACTIONS):
                    g_copy = deepcopy(g)
                    s, _, _, _, _ = g_copy.step([i, 99])  
                    if evalFunctions.n_defeated(s[0].teams[0]) > evalFunctions.n_defeated(g.teams[0]): 
                        continue 
                    eval_score, _ = self.minimax(s[0], depth - 1, False)
                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_action = i
            except Exception as e:
                import traceback
                traceback.print_exc()

            return max_eval, best_action

        else: #MIN
            min_eval = float('inf')
            best_action = None
            try:
                for j in range(DEFAULT_N_ACTIONS):
                    g_copy = deepcopy(g)
                    s, _, _, _, _ = g_copy.step([99, j])  
                    if evalFunctions.n_defeated(s[0].teams[1]) > evalFunctions.n_defeated(g.teams[1]):
                        continue
                    eval_score, _ = self.minimax(s[0], depth - 1, True)
                    if eval_score < min_eval:
                        min_eval = eval_score
                        best_action = j
            except Exception as e:
                import traceback
                traceback.print_exc()

            return min_eval, best_action


    def get_action(self, g) -> int:
        """
        Find the best action for MAX
        """
        try:
            _, best_action = self.minimax(g, self.max_depth, True)
        except Exception as e:
            import traceback
            traceback.print_exc()

        return best_action if best_action is not None else 0

