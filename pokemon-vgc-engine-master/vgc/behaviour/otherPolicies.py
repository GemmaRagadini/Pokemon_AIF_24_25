import math
import hashlib
import numpy as np
import random
from typing import List
from vgc.behaviour import evalFunctions
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition, PkmEntryHazard
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
        Algoritmo Minimax classico.
        """

        if depth == 0:
            # Valuta lo stato corrente.
            try:
                evaluation = evalFunctions.my_eval_fun(g, depth)
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            return evaluation , None

        if is_maximizing_player:
            max_eval = float('-inf')
            best_action = None
            try:
                for i in range(DEFAULT_N_ACTIONS):
                    g_copy = deepcopy(g)
                    s, _, _, _, _ = g_copy.step([i, 99])  # L'avversario esegue un'azione non valida,che non cambia la situazione
                    if evalFunctions.n_fainted(s[0].teams[0]) > evalFunctions.n_fainted(g.teams[0]): 
                        continue # Ignora gli stati in cui il nostro numero di Pokémon sconfitti aumenta.
                    eval_score, _ = self.minimax(s[0], depth - 1, False)
                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_action = i
            except Exception as e:
                import traceback
                traceback.print_exc()

            return max_eval, best_action

        else:  # Avversario minimizzante
            min_eval = float('inf')
            best_action = None
            try:
                for j in range(DEFAULT_N_ACTIONS):
                    g_copy = deepcopy(g)
                    s, _, _, _, _ = g_copy.step([99, j])  # Il giocatore non cambia azione (azione non valida)
                    # Ignora gli stati in cui il numero di Pokémon sconfitti dell'avversario aumenta.
                    if evalFunctions.n_fainted(s[0].teams[1]) > evalFunctions.n_fainted(g.teams[1]):
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
        Trova la migliore azione da intraprendere per il giocatore massimizzante.
        """
        try:
            _, best_action = self.minimax(g, self.max_depth, True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            
        return best_action if best_action is not None else 0



class Node:
    def __init__(self, state, parent, player, action=None):
        self.state = state
        self.parent = parent
        self.player = player
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.get_untried_actions()) == 0

    def get_untried_actions(self):
        # Ottiene le azioni possibili dallo stato
        return [i for i in range(DEFAULT_N_ACTIONS)]


class MyMinimaxWithAlphaBetaKiller(BattlePolicy):

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.name = "Minimax with pruning alpha beta killer"
        self.killer_moves = {depth: [] for depth in range(max_depth + 1)}  # Memorizza le killer moves per profondità

    def minimax(self, g, depth, alpha, beta, is_maximizing_player):
        if depth == 0:
            return evalFunctions.game_state_eval(g, depth), None

        if is_maximizing_player:
            max_eval = float('-inf')
            best_action = None

            # Ottieni le azioni disponibili
            moves = list(range(DEFAULT_N_ACTIONS))

            # Prioritizza le killer moves
            killer_moves = self.killer_moves.get(depth, [])
            moves = sorted(moves, key=lambda move: move in killer_moves, reverse=True)

            for i in moves:
                g_copy = deepcopy(g)
                s, _, _, _, _ = g_copy.step([i, 99])
                if evalFunctions.n_fainted(s[0].teams[0]) > evalFunctions.n_fainted(g.teams[0]):
                    continue

                eval_score, _ = self.minimax(s[0], depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = i

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    # Aggiorna le killer moves
                    if i not in self.killer_moves[depth]:
                        self.killer_moves[depth].append(i)
                        if len(self.killer_moves[depth]) > 2:  
                            self.killer_moves[depth].pop(0)
                    break
            return max_eval, best_action

        else:
            min_eval = float('inf')
            best_action = None

            # Ottieni le azioni disponibili
            moves = list(range(DEFAULT_N_ACTIONS))

            # Prioritizza le killer moves
            killer_moves = self.killer_moves.get(depth, [])
            moves = sorted(moves, key=lambda move: move in killer_moves, reverse=True)

            for j in moves:
                g_copy = deepcopy(g)
                s, _, _, _, _ = g_copy.step([99, j])
                if evalFunctions.n_fainted(s[0].teams[1]) > evalFunctions.n_fainted(g.teams[1]):
                    continue

                eval_score, _ = self.minimax(s[0], depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = j

                beta = min(beta, eval_score)
                if beta <= alpha:
                    # Aggiorna le killer moves
                    if j not in self.killer_moves[depth]:
                        self.killer_moves[depth].append(j)
                        if len(self.killer_moves[depth]) > 2: 
                            self.killer_moves[depth].pop(0)
                    break
            return min_eval, best_action

    def get_action(self, g) -> int:
        _, best_action = self.minimax(g, self.max_depth, float('-inf'), float('inf'), True)
        return best_action if best_action is not None else 0




class MCTS_MS(BattlePolicy):
    def __init__(self, max_iterations: int = 1000, exploration_weight: float = 1.41, minimax_depth: int = 4):
        """
        Monte Carlo Tree Search con Minimax Selection (MCTS-MS).
        """
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight
        self.minimax_depth = minimax_depth
        self.name = "Monte Carlo with Minimax Selection"

    def mcts(self, g, root_player: int):
        # Nodo radice
        root = Node(state=deepcopy(g), parent=None, player=root_player)

        for _ in range(self.max_iterations):
            leaf = self.select(root)
            simulation_result = self.simulate(leaf.state, leaf.player)
            self.backpropagate(leaf, simulation_result)

        # Scegli la migliore azione dalla radice
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.action

    def select(self, node):
        """
        Seleziona un nodo utilizzando UCT e Minimax Selection.

        """
        while node.children:
            # Applica minimax se un nodo raggiunge un numero minimo di visite
            if node.visits >= 5:  
                minimax_result = self.minimax(node.state, self.minimax_depth, node.player)
                if minimax_result is not None:
                    return Node(state=deepcopy(node.state), parent=node, player=node.player)

            # Se non soddisfa i criteri di minimax, usa UCT per scegliere
            node = max(node.children, key=lambda child: self.uct(child))
        
        # continua ad espandere il nodo
        if not node.is_fully_expanded():
            return self._expand(node)
        return node

    def minimax(self, g, depth, player):
        """
        Usa una ricerca Minimax a profondità limitata per valutare uno stato.
        """
        def minimax(g, depth, is_maximizing):
            if depth == 0 or g.is_terminal():
                return evalFunctions.game_state_eval(g, player)

            if is_maximizing:
                max_eval = float('-inf')
                for i in range(DEFAULT_N_ACTIONS):
                    g_copy = deepcopy(g)
                    s, _, _, _, _ = g_copy.step([i, 99])
                    eval_score = minimax(s[0], depth - 1, False)
                    max_eval = max(max_eval, eval_score)
                return max_eval
            else:
                min_eval = float('inf')
                for j in range(DEFAULT_N_ACTIONS):
                    g_copy = deepcopy(g)
                    s, _, _, _, _ = g_copy.step([99, j])
                    eval_score = minimax(s[0], depth - 1, True)
                    min_eval = min(min_eval, eval_score)
                return min_eval

        eval_score = minimax(g, depth, True)
        return eval_score if eval_score > 0.5 else None  

    def _expand(self, node):
        untried_actions = node.get_untried_actions()
        action = random.choice(untried_actions)
        new_state, _, _, _, _ = deepcopy(node.state).step([action, 99])
        child_node = Node(state=new_state, parent=node, player=1 - node.player, action=action)
        node.children.append(child_node)
        return child_node

    def simulate(self, state, player):
        g_copy = deepcopy(state)
        while not g_copy.is_terminal():
            action = random.randint(0, DEFAULT_N_ACTIONS - 1)
            g_copy.step([action, 99]) 
        return self.evaluate(g_copy, player)

    def evaluate(self, state, player):
        return evalFunctions.game_state_eval(state, player)

    def backpropagate(self, node, result):
        while node:
            node.visits += 1
            node.value += result if node.player == node.parent.player else -result
            node = node.parent

    def uct(self, node):
        if node.visits == 0:
            return float('inf')
        return (node.value / node.visits +
                self.exploration_weight * math.sqrt(math.log(node.parent.visits) / node.visits))

    def get_action(self, g) -> int:
        return self.mcts(g, root_player=0)




