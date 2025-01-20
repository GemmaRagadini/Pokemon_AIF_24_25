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
        Algoritmo Minimax classico.
        """

        if depth == 0:
            # Valuta lo stato corrente.
            try:
                evaluation = evalFunctions.my_eval_fun(g,depth)
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



class MyMinimaxWithAlphaBetaKiller(BattlePolicy):

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.name = "Minimax with pruning alpha beta killer"
        self.killer_moves = {depth: [] for depth in range(max_depth + 1)}  # Memorizza le killer moves per profondità

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

            # Ottieni le azioni disponibili
            moves = list(range(DEFAULT_N_ACTIONS))

            # Prioritizza le killer moves
            killer_moves = self.killer_moves.get(depth, [])
            moves = sorted(moves, key=lambda move: move in killer_moves, reverse=True)

            try:
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
            except Exception as e:
                import traceback
                traceback.print_exc()

            return max_eval, best_action

        else:
            min_eval = float('inf')
            best_action = None

            # Ottieni le azioni disponibili
            moves = list(range(DEFAULT_N_ACTIONS))

            try:
                # Prioritizza le killer moves
                killer_moves = self.killer_moves.get(depth, [])
                moves = sorted(moves, key=lambda move: move in killer_moves, reverse=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            try:
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



# class MCTS_MS(BattlePolicy):

#     def __init__(self):
#         self.hail_used = False
#         self.sandstorm_used = False
#         self.name = "Monte Carlo with Minimax Selection"

#     def estimate_damages(self, active_pkm: Pkm, opp_pkm_type: PkmType, attack_stage: int, defense_stage: int, weather: WeatherCondition) -> int:
#         # valutazione mosse
#         damages: List[float] = []
#         for move in active_pkm.moves:
#             damages.append(evalFunctions.estimate_damage(move.type, active_pkm.type, move.power, opp_pkm_type, attack_stage,
#                                           defense_stage, weather))
#         return damages

#     def simulate_action(self, game_state: GameState, action: int) -> float:
#         """
#         Simula l'azione scelta e restituisce una stima del punteggio risultante.
#         """
#         simulated_state = game_state.clone()
#         if action < 4:  # Azione di attacco
#             simulated_state.apply_move(action)
#         else:  # Cambio Pokémon
#             simulated_state.switch_pokemon(action - 4)
        
#         # Una valutazione semplice del punteggio dello stato (da perfezionare con metriche più complesse)
#         return evalFunctions.evaluate_game_state(simulated_state)

#     def monte_carlo_minimax(self, game_state: GameState, depth: int, is_maximizing: bool) -> float:
#         """
#         Algoritmo Monte Carlo con selezione Minimax.
        
#         - depth: profondità della simulazione
#         - is_maximizing: True se è il turno dell'agente, False altrimenti
        
#         Restituisce il miglior punteggio stimato.
#         """
#         if depth == 0 or game_state.is_terminal():
#             return evalFunctions.evaluate_game_state(game_state)

#         # Ottieni tutte le azioni possibili (4 mosse + cambi Pokémon)
#         possible_actions = list(range(4))  # Mosse
#         bench = game_state.teams[0].party
#         for i, pkm in enumerate(bench):
#             if pkm.hp > 0:  # Solo Pokémon non esausti
#                 possible_actions.append(4 + i)
        
#         if is_maximizing:
#             max_eval = float('-inf')
#             for action in possible_actions:
#                 # Simula e valuta l'azione
#                 simulated_state = game_state.clone()
#                 if action < 4:
#                     simulated_state.apply_move(action)
#                 else:
#                     simulated_state.switch_pokemon(action - 4)
#                 eval_score = self.monte_carlo_minimax(simulated_state, depth - 1, False)
#                 max_eval = max(max_eval, eval_score)
#             return max_eval
#         else:
#             min_eval = float('inf')
#             for action in possible_actions:
#                 # Simula e valuta l'azione
#                 simulated_state = game_state.clone()
#                 if action < 4:
#                     simulated_state.apply_move(action)
#                 else:
#                     simulated_state.switch_pokemon(action - 4)
#                 eval_score = self.monte_carlo_minimax(simulated_state, depth - 1, True)
#                 min_eval = min(min_eval, eval_score)
#             return min_eval

#     def get_action(self, g: GameState) -> int:
#         # la mia squadra
#         my_team = g.teams[0]
#         active_pkm = my_team.active
#         bench = my_team.party
#         my_attack_stage = my_team.stage[PkmStat.ATTACK]

#         # squadra avversaria
#         opp_team = g.teams[1]
#         opp_active_pkm = opp_team.active
#         opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

#         # meteo 
#         weather = g.weather.condition

#         try:
#             # stima dei danni di ogni mossa
#             damages = self.estimate_damages(active_pkm, opp_active_pkm.type, my_attack_stage, opp_defense_stage, weather)
#             move_id = int(np.argmax(damages))
#         except Exception as e:
#             import traceback
#             traceback.print_exc()

#         # Seleziona la miglior azione usando Monte Carlo Minimax
#         best_action = -1
#         best_score = float('-inf')
        
#         try:
#             # Azioni disponibili (4 mosse + cambi Pokémon)
#             possible_actions = list(range(4))  # Mosse
#             for i, pkm in enumerate(bench):
#                 if pkm.hp > 0:  # Solo Pokémon non esausti
#                     possible_actions.append(4 + i)

#             for action in possible_actions:
#                 simulated_score = self.monte_carlo_minimax(g, depth=3, is_maximizing=True)
#                 if simulated_score > best_score:
#                     best_score = simulated_score
#                     best_action = action
#         except Exception as e:
#             import traceback
#             traceback.print_exc()

#         return best_action
