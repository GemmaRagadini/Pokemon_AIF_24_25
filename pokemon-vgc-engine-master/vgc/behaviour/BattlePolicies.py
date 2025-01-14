import tkinter
from copy import deepcopy
from threading import Thread, Event
from tkinter import CENTER, DISABLED, NORMAL
from types import CellType
from typing import List
from random import random
import numpy as np
from customtkinter import CTk, CTkButton, CTkRadioButton, CTkLabel

from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER, DEFAULT_N_ACTIONS
from vgc.datatypes.Objects import GameState, PkmTeam
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition
import math
import hashlib

class RandomPlayer(BattlePolicy):
    """
    Agent that selects actions randomly.
    """

    def __init__(self, switch_probability: float = .15, n_moves: int = DEFAULT_PKM_N_MOVES,
                 n_switches: int = DEFAULT_PARTY_SIZE):
        super().__init__()
        self.n_actions: int = n_moves + n_switches
        self.pi: List[float] = ([(1. - switch_probability) / n_moves] * n_moves) + (
                [switch_probability / n_switches] * n_switches)

    def get_action(self, g: GameState) -> int:
        return np.random.choice(self.n_actions, p=self.pi)


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


class OneTurnLookahead(BattlePolicy):
    """
    Greedy heuristic based competition designed to encapsulate a greedy strategy that prioritizes damage output.
    Source: http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf
    """

    def get_action(self, g: GameState):
        # get weather condition
        weather = g.weather.condition

        # get my pkms
        my_team = g.teams[0]
        my_active = my_team.active
        my_attack_stage = my_team.stage[PkmStat.ATTACK]

        # get opp team
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_active_type = opp_active.type
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        # get most damaging move from my active pkm
        damage: List[float] = []
        for move in my_active.moves:
            damage.append(estimate_damage(move.type, my_active.type, move.power, opp_active_type,
                                          my_attack_stage, opp_defense_stage, weather))

        return int(np.argmax(damage))  # use active pkm best damaging move


def match_up_eval(my_pkm_type: PkmType, opp_pkm_type: PkmType, opp_moves_type: List[PkmType]) -> float:
    # determine defensive match up
    defensive_match_up = 0.0
    for mtype in opp_moves_type + [opp_pkm_type]:
        defensive_match_up = max(TYPE_CHART_MULTIPLIER[mtype][my_pkm_type], defensive_match_up)
    return defensive_match_up


class TypeSelector(BattlePolicy):
    """
    Type Selector is a variation upon the One Turn Lookahead competition that utilizes a short series of if-else
    statements in its decision-making.
    Source: http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf
    """

    def get_action(self, g: GameState):
        # get weather condition
        weather = g.weather.condition

        # get my pokémon
        my_team = g.teams[0]
        my_active = my_team.active
        my_party = my_team.party
        my_attack_stage = my_team.stage[PkmStat.ATTACK]

        # get opp team
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_active_type = opp_active.type
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        # get most damaging move from my active pokémon
        damage: List[float] = []
        for move in my_active.moves:
            damage.append(estimate_damage(move.type, my_active.type, move.power, opp_active_type,
                                          my_attack_stage, opp_defense_stage, weather))
        move_id = int(np.argmax(damage))

        #  If this damage is greater than the opponents current health we knock it out
        if damage[move_id] >= opp_active.hp:
            return move_id

        # If not, check if are a favorable match. If we are lets give maximum possible damage.
        if match_up_eval(my_active.type, opp_active.type, list(map(lambda m: m.type, opp_active.moves))) <= 1.0:
            return move_id

        # If we are not switch to the most favorable match up
        match_up: List[float] = []
        not_fainted = False
        for pkm in my_party:
            if pkm.hp == 0.0:
                match_up.append(2.0)
            else:
                not_fainted = True
                match_up.append(
                    match_up_eval(pkm.type, opp_active.type, list(map(lambda m: m.type, opp_active.moves))))

        if not_fainted:
            return int(np.argmin(match_up)) + 4

        # If our party has no non fainted pkm, lets give maximum possible damage with current active
        return move_id


class BFSNode:

    def __init__(self):
        self.a = None
        self.g = None
        self.parent = None
        self.depth = 0
        self.eval = 0.0


def n_fainted(t: PkmTeam):
    fainted = 0
    fainted += t.active.hp == 0
    if len(t.party) > 0:
        fainted += t.party[0].hp == 0
    if len(t.party) > 1:
        fainted += t.party[1].hp == 0
    return fainted


def game_state_eval(s: GameState, depth):
    mine = s.teams[0].active
    opp = s.teams[1].active
    return mine.hp / mine.max_hp - 3 * opp.hp / opp.max_hp - 0.3 * depth

# def game_state_eval(s: GameState, depth):
#     mine = s.teams[0].active
#     opp = s.teams[1].active
#     # Punti vita relativi (bilanciati)
#     hp_score = (mine.hp / mine.max_hp) - (opp.hp / opp.max_hp)
#     # Penalità per profondità (ricompensa azioni rapide)
#     depth_penalty = -0.3 * depth
#     # Considera i Pokémon svenuti (vantaggio numerico)
#     fainted_score = len(s.teams[1].bench) - len(s.teams[0].bench)
#     # Punteggio finale con pesi aggiustati
#     return hp_score + 0.5 * fainted_score + depth_penalty



class BreadthFirstSearch(BattlePolicy):
    """
    Basic tree search algorithm that traverses nodes in level order until it finds a state in which the current opponent
    Pokémon is fainted. As a non-adversarial algorithm, the competition selfishly assumes that the opponent uses
    "force skip" (by selecting an invalid switch action).
    Source: http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf
    """

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth

    def get_action(self, g) -> int:  # g: PkmBattleEnv
        root: BFSNode = BFSNode()
        root.g = g
        node_queue: List[BFSNode] = [root]
        while len(node_queue) > 0 and node_queue[0].depth < self.max_depth:
            current_parent = node_queue.pop(0)
            # expand nodes of current parent
            for i in range(DEFAULT_N_ACTIONS):
                g = deepcopy(current_parent.g)
                s, _, _, _, _ = g.step([i, 99])  # opponent select an invalid switch action
                # our fainted increased, skip
                if n_fainted(s[0].teams[0]) > n_fainted(current_parent.g.teams[0]):
                    continue
                # our opponent fainted increased, follow this decision
                if n_fainted(s[0].teams[1]) > n_fainted(current_parent.g.teams[1]):
                    a = i
                    while current_parent != root:
                        a = current_parent.a
                        current_parent = current_parent.parent
                    return a
                # continue tree traversal
                node = BFSNode()
                node.parent = current_parent
                node.depth = node.parent.depth + 1
                node.a = i
                node.g = s[0]
                node_queue.append(node)
        # no possible win outcomes, return arbitrary action
        if len(node_queue) == 0:
            return 0
        # return action with most potential
        best_node = max(node_queue, key=lambda n: game_state_eval(n.g, n.depth))
        while best_node.parent != root:
            best_node = best_node.parent
        return best_node.a


class Minimax(BattlePolicy):
    """
    Tree search algorithm that deals with adversarial paradigms by assuming the opponent acts in their best interest.
    Each node in this tree represents the worst case scenario that would occur if the player had chosen a specific
    choice.
    Source: http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf
    """

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth

    def get_action(self, g) -> int:  # g: PkmBattleEnv
        root: BFSNode = BFSNode()
        root.g = g
        node_queue: List[BFSNode] = [root]
        while len(node_queue) > 0 and node_queue[0].depth < self.max_depth:
            current_parent = node_queue.pop(0)
            # expand nodes of current parent
            for i in range(DEFAULT_N_ACTIONS):
                for j in range(DEFAULT_N_ACTIONS):
                    g = deepcopy(current_parent.g)
                    s, _, _, _, _ = g.step([i, j])  # opponent select an invalid switch action
                    # our fainted increased, skip
                    if n_fainted(s[0].teams[0]) > n_fainted(current_parent.g.teams[0]):
                        continue
                    # our opponent fainted increased, follow this decision
                    if n_fainted(s[0].teams[1]) > n_fainted(current_parent.g.teams[1]):
                        a = i
                        while current_parent != root:
                            a = current_parent.a
                            current_parent = current_parent.parent
                        return a
                    # continue tree traversal
                    node = BFSNode()
                    node.parent = current_parent
                    node.depth = node.parent.depth + 1
                    node.a = i
                    node.g = s[0]
                    node_queue.append(node)
        # no possible win outcomes, return arbitrary action
        if len(node_queue) == 0:
            return 0
        # return action with most potential
        best_node = max(node_queue, key=lambda n: game_state_eval(n.g, n.depth))
        while best_node.parent != root:
            best_node = best_node.parent
        return best_node.a


class PrunedBFS(BattlePolicy):
    """
    Utilize domain knowledge as a cost-cutting measure by making modifications to the Breadth First Search competition.
    We do not simulate any actions that involve using a damaging move with a resisted type, nor does it simulate any
    actions that involve switching to a Pokémon with a subpar type match up. Additionally, rather than selfishly
    assuming the opponent skips their turn in each simulation, the competition assumes its opponent is a
    One Turn Lookahead competition.
    Source: http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf
    """

    def __init__(self, max_depth: int = 4):
        self.core_agent: BattlePolicy = OneTurnLookahead()
        self.max_depth = max_depth

    def get_action(self, g) -> int:  # g: PkmBattleEnv
        root: BFSNode = BFSNode()
        root.g = g
        node_queue: List[BFSNode] = [root]
        while len(node_queue) > 0 and node_queue[0].depth < self.max_depth:
            current_parent = node_queue.pop(0)
            # assume opponent follows just the OneTurnLookahead strategy, which is more greedy in damage
            o: GameState = deepcopy(current_parent.g)
            # opponent must see the teams swapped
            o.teams = (o.teams[1], o.teams[0])
            j = self.core_agent.get_action(o)
            # expand nodes
            for i in range(DEFAULT_N_ACTIONS):
                g = deepcopy(current_parent.g)
                my_team = g.teams[0]
                my_active = my_team.active
                opp_team = g.teams[1]
                opp_active = opp_team.active
                # skip traversing tree with non very effective moves
                if i < 4 and TYPE_CHART_MULTIPLIER[my_active.moves[i].type][opp_active.type] < 1.0:
                    continue
                # skip traversing tree with switches to Pokémon that are a bad type match against opponent active
                elif i >= 4:
                    p = i - DEFAULT_N_ACTIONS
                    for move in opp_active.moves:
                        if move.power > 0.0 and TYPE_CHART_MULTIPLIER[move.type][my_team.party[p].type] > 1.0:
                            continue
                s, _, _, _, _ = g.step([i, j])
                # our fainted increased, skip
                if n_fainted(s[0].teams[0]) > n_fainted(current_parent.g.teams[0]):
                    continue
                # our opponent fainted increased, follow this decision
                if n_fainted(s[0].teams[1]) > n_fainted(current_parent.g.teams[1]):
                    a = i
                    while current_parent != root:
                        a = current_parent.a
                        current_parent = current_parent.parent
                    return a
                # continue tree traversal
                node = BFSNode()
                node.parent = current_parent
                node.depth = node.parent.depth + 1
                node.a = i
                node.g = s[0]
                node_queue.append(node)
        # no possible win outcomes, return arbitrary action
        if len(node_queue) == 0:
            a = self.core_agent.get_action(g)
            return a
        # return action with most potential
        best_node = max(node_queue, key=lambda n: game_state_eval(n.g, n.depth))
        while best_node.parent != root:
            best_node = best_node.parent
        return best_node.a


class TunedTreeTraversal(BattlePolicy):
    """
    Agent inspired on PrunedBFS. Assumes opponent is a TypeSelector. It traverses only to states after using moves
    recommended by a TypeSelector agent and non-damaging moves.
    """

    def __init__(self, max_depth: int = 4):
        self.core_agent: BattlePolicy = TypeSelector()
        self.max_depth = max_depth

    def get_action(self, g: GameState) -> int:  # g: PkmBattleEnv
        root: BFSNode = BFSNode()
        root.g = g
        node_queue: List[BFSNode] = [root]
        while len(node_queue) > 0 and node_queue[0].depth < self.max_depth:
            current_parent = node_queue.pop(0)
            # assume opponent follows just the TypeSelector strategy, which is more greedy in damage
            o: GameState = deepcopy(current_parent.g)
            # opponent must see the teams swapped
            o.teams = (o.teams[1], o.teams[0])
            j = self.core_agent.get_action(o)
            # expand nodes with TypeSelector strategy plus non-damaging moves
            for i in [self.core_agent.get_action(current_parent.g)] + [i for i, m in enumerate(
                    current_parent.g.teams[0].active.moves) if m.power == 0.]:
                g = deepcopy(current_parent.g)
                s, _, _, _, _ = g.step([i, j])
                # our fainted increased, skip
                if n_fainted(s[0].teams[0]) > n_fainted(current_parent.g.teams[0]):
                    continue
                # our opponent fainted increased, follow this decision
                if n_fainted(s[0].teams[1]) > n_fainted(current_parent.g.teams[1]):
                    a = i
                    while current_parent != root:
                        a = current_parent.a
                        current_parent = current_parent.parent
                    return a
                # continue tree traversal
                node = BFSNode()
                node.parent = current_parent
                node.depth = node.parent.depth + 1
                node.a = i
                node.g = s[0]
                node_queue.append(node)
        # no possible win outcomes, return arbitrary action
        if len(node_queue) == 0:
            a = self.core_agent.get_action(g)
            return a
        # return action with most potential
        best_node = max(node_queue, key=lambda n: game_state_eval(n.g, n.depth))
        while best_node.parent != root:
            best_node = best_node.parent
        return best_node.a


class TerminalPlayer(BattlePolicy):
    """
    Terminal interface.
    """

    def get_action(self, g: GameState) -> int:
        print('~ Actions ~')
        for i, a in enumerate(g.teams[0].active.moves):
            print(i, '->', a)
        for i, a in enumerate(g.teams[0].party):
            print(i + DEFAULT_PKM_N_MOVES, '-> Switch to ', a)
        while True:
            try:
                m = int(input('Select Action: '))
                if 0 < m < DEFAULT_N_ACTIONS:
                    break
                else:
                    print('Invalid action. Select again.')
            except:
                print('Invalid action. Select again.')
        print()
        return m


def run_tk(event, value_wrapper, game_state_wrapper):
    app = GUIPlayer.App()
    app.event = event
    app.value_wrapper = value_wrapper
    app.game_state_wrapper = game_state_wrapper
    app.update_game_state()
    app.mainloop()


def disable_event():
    pass


class GUIPlayer(BattlePolicy):
    """
    Graphical interface.
    """

    class App(CTk):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.title("VGC Battle GUI")
            #self.iconbitmap(r"vgc/ux/vgc_v2_01_Uw5_icon.ico")
            self.geometry("650x200")
            self.protocol("WM_DELETE_WINDOW", disable_event)
            CTkLabel(self, text="Actions").pack(anchor='w')
            self.button = CTkButton(self, text="Select Action", state=DISABLED, command=self.button_callback)
            self.button.place(relx=0.5, rely=0.9, anchor=CENTER)
            self.radio_var = tkinter.IntVar(value=0)  # Create a variable for strings, and initialize the variable
            self.buttons = []
            for i in range(DEFAULT_N_ACTIONS):
                button = CTkRadioButton(self, text=f"{i}", state=DISABLED, variable=self.radio_var, value=i)
                button.pack(anchor='w')
                self.buttons += [button]
            self.event = None
            self.value_wrapper = None
            self.game_state_wrapper = None

        def button_callback(self):
            self.button.configure(state=DISABLED)
            for button in self.buttons:
                button.configure(state=DISABLED)
            self.value_wrapper.cell_contents = self.radio_var.get()
            self.event.set()

        def update_game_state(self):
            g = self.game_state_wrapper.cell_contents
            if g is not None and not isinstance(g, int):
                self.button.configure(state=NORMAL)
                for i, button in enumerate(self.buttons):
                    if i < DEFAULT_PKM_N_MOVES:
                        button.configure(state=NORMAL, text=f"{g.teams[0].active.moves[i]}")
                    else:
                        button.configure(state=NORMAL, text=f"Switch to {g.teams[0].party[i - DEFAULT_PKM_N_MOVES]}")
                self.game_state_wrapper.cell_contents = None
            self.after(1000, self.update_game_state)

    def __init__(self):
        self.event = Event()
        self.value_wrapper = CellType(0)
        self.game_state_wrapper = CellType(0)
        self.thd = Thread(target=run_tk, args=(self.event, self.value_wrapper, self.game_state_wrapper))
        self.thd.daemon = True
        self.thd.start()

    def get_action(self, g: GameState) -> int:
        self.game_state_wrapper.cell_contents = g
        self.event.wait()
        self.event.clear()
        return self.value_wrapper.cell_contents


##################################

class MyMinimax(BattlePolicy):
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.name = "Minimax"

    def minimax(self, g, depth, is_maximizing_player):
        """
        Algoritmo Minimax classico.
        
        :param g: Lo stato corrente del gioco.
        :param depth: La profondità corrente nella ricerca.
        :param is_maximizing_player: True se il giocatore corrente è MAX, False se è MIN.
        :return: (valutazione, azione migliore)
        """
        if depth == 0:
            # Valuta lo stato corrente.
            return game_state_eval(g, depth), None

        if is_maximizing_player:
            max_eval = float('-inf')
            best_action = None
            for i in range(DEFAULT_N_ACTIONS):
                g_copy = deepcopy(g)
                s, _, _, _, _ = g_copy.step([i, 99])  # L'avversario esegue un'azione non valida,che non cambia la situazione
                if n_fainted(s[0].teams[0]) > n_fainted(g.teams[0]): 
                    continue # Ignora gli stati in cui il nostro numero di Pokémon sconfitti aumenta.
                eval_score, _ = self.minimax(s[0], depth - 1, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = i
            return max_eval, best_action

        else:  # Avversario minimizzante
            min_eval = float('inf')
            best_action = None
            for j in range(DEFAULT_N_ACTIONS):
                g_copy = deepcopy(g)
                s, _, _, _, _ = g_copy.step([99, j])  # Il giocatore non cambia azione (azione non valida)
                # Ignora gli stati in cui il numero di Pokémon sconfitti dell'avversario aumenta.
                if n_fainted(s[0].teams[1]) > n_fainted(g.teams[1]):
                    continue
                eval_score, _ = self.minimax(s[0], depth - 1, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = j
            return min_eval, best_action

    def get_action(self, g) -> int:
        """
        Trova la migliore azione da intraprendere per il giocatore massimizzante.
        
        :param g: Lo stato corrente del gioco.
        :return: L'azione migliore da eseguire.
        """
        _, best_action = self.minimax(g, self.max_depth, True)
        return best_action if best_action is not None else 0




class MyMinimaxWithAlphaBeta(BattlePolicy):
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.name = "Minimax with pruning alpha beta"

    def minimax(self, g, depth, alpha, beta, is_maximizing_player):
        """
        Algoritmo Minimax con Alpha-Beta Pruning.

        :param g: Lo stato corrente del gioco.
        :param depth: La profondità corrente nella ricerca.
        :param alpha: Il valore massimo che il giocatore MAX garantisce finora.
        :param beta: Il valore minimo che il giocatore MIN garantisce finora.
        :param is_maximizing_player: True se il giocatore corrente è MAX, False se è MIN.
        :return: (valutazione, azione migliore)
        """
        if depth == 0:
            # Valuta lo stato corrente.
            return game_state_eval(g, depth), None

        if is_maximizing_player:
            max_eval = float('-inf')
            best_action = None
            for i in range(DEFAULT_N_ACTIONS):
                g_copy = deepcopy(g)
                s, _, _, _, _ = g_copy.step([i, 99])  # L'avversario esegue un'azione non valida.
                if n_fainted(s[0].teams[0]) > n_fainted(g.teams[0]): 
                    continue  # Ignora stati in cui i nostri Pokémon sconfitti aumentano.

                eval_score, _ = self.minimax(s[0], depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = i
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Potatura beta
            return max_eval, best_action

        else:  # Avversario minimizzante
            min_eval = float('inf')
            best_action = None
            for j in range(DEFAULT_N_ACTIONS):
                g_copy = deepcopy(g)
                s, _, _, _, _ = g_copy.step([99, j])  # Il giocatore non cambia azione.
                if n_fainted(s[0].teams[1]) > n_fainted(g.teams[1]):
                    continue  # Ignora stati in cui i Pokémon sconfitti dell'avversario aumentano.

                eval_score, _ = self.minimax(s[0], depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = j
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Potatura alpha
            return min_eval, best_action

    def get_action(self, g) -> int:
        """
        Trova la migliore azione da intraprendere per il giocatore massimizzante.

        :param g: Lo stato corrente del gioco.
        :return: L'azione migliore da eseguire.
        """
        _, best_action = self.minimax(g, self.max_depth, float('-inf'), float('inf'), True)
        return best_action if best_action is not None else 0



class MyMonteCarlo(BattlePolicy):
    def __init__(self, max_iterations: int = 1000, exploration_weight: float = 1.41):
        """
        Inizializza la politica Monte Carlo Tree Search (MCTS).

        :param max_iterations: Numero massimo di iterazioni per MCTS.
        :param exploration_weight: Peso per il termine di esplorazione (valore di C).
        """
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight
        self.name = "Monte Carlo"

    def mcts(self, g, root_player: int):
        """
        Implementa il processo di Monte Carlo Tree Search.

        :param g: Lo stato corrente del gioco.
        :param root_player: Il giocatore per cui stiamo cercando la migliore azione.
        :return: La migliore azione da eseguire.
        """
        # Nodo radice
        root = Node(state=deepcopy(g), parent=None, player=root_player)

        for _ in range(self.max_iterations):
            leaf = self._select(root)
            simulation_result = self._simulate(leaf.state, leaf.player)
            self._backpropagate(leaf, simulation_result)

        # Scegli la migliore azione dalla radice
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.action

    def _select(self, node):
        """
        Seleziona un nodo utilizzando UCT (Upper Confidence Bound for Trees).

        :param node: Il nodo corrente.
        :return: Il nodo foglia selezionato.
        """
        while node.children:
            node = max(node.children, key=lambda child: self._uct(child))
        if not node.is_fully_expanded():
            return self._expand(node)
        return node

    def _expand(self, node):
        """
        Espande il nodo generando un nuovo figlio.

        :param node: Il nodo corrente.
        :return: Il nuovo nodo figlio.
        """
        untried_actions = node.get_untried_actions()
        action = random.choice(untried_actions)
        new_state, _, _, _, _ = deepcopy(node.state).step([action, 99])  # Azione avversario non valida
        child_node = Node(state=new_state, parent=node, player=1 - node.player, action=action)
        node.children.append(child_node)
        return child_node

    def _simulate(self, state, player):
        """
        Esegue una simulazione casuale a partire da uno stato.

        :param state: Lo stato iniziale della simulazione.
        :param player: Il giocatore corrente.
        :return: Il risultato della simulazione.
        """
        g_copy = deepcopy(state)
        while not g_copy.is_terminal():
            action = random.randint(0, DEFAULT_N_ACTIONS - 1)
            g_copy.step([action, 99])  # Simulazione casuale
        return self._evaluate(g_copy, player)

    def _evaluate(self, state, player):
        """
        Valuta lo stato finale per un giocatore specifico.

        :param state: Lo stato finale.
        :param player: Il giocatore da valutare.
        :return: Un valore di ricompensa per il giocatore.
        """
        return game_state_eval(state, player)

    def _backpropagate(self, node, result):
        """
        Propaga i risultati della simulazione verso la radice.

        :param node: Il nodo corrente.
        :param result: Il risultato della simulazione.
        """
        while node:
            node.visits += 1
            node.value += result if node.player == node.parent.player else -result
            node = node.parent

    def _uct(self, node):
        """
        Calcola l'Upper Confidence Bound per un nodo.

        :param node: Il nodo per cui calcolare UCT.
        :return: Il valore UCT del nodo.
        """
        if node.visits == 0:
            return float('inf')  # Esplora nodi non visitati
        return (node.value / node.visits +
                self.exploration_weight * math.sqrt(math.log(node.parent.visits) / node.visits))

    def get_action(self, g) -> int:
        """
        Trova la migliore azione utilizzando MCTS.

        :param g: Lo stato corrente del gioco.
        :return: L'azione migliore da eseguire.
        """
        return self.mcts(g, root_player=0)

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

#si prova con la killer move


class MyMinimaxWithAlphaBetaKiller(BattlePolicy):

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.name = "Minimax with pruning alpha beta killer"
        self.killer_moves = {depth: [] for depth in range(max_depth + 1)}  # Memorizza le killer moves per profondità

    def minimax(self, g, depth, alpha, beta, is_maximizing_player):
        if depth == 0:
            return game_state_eval(g, depth), None

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
                if n_fainted(s[0].teams[0]) > n_fainted(g.teams[0]):
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
                        if len(self.killer_moves[depth]) > 2:  # Limita a 2 killer moves per profondità
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
                if n_fainted(s[0].teams[1]) > n_fainted(g.teams[1]):
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
                        if len(self.killer_moves[depth]) > 2:  # Limita a 2 killer moves per profondità
                            self.killer_moves[depth].pop(0)
                    break
            return min_eval, best_action

    def get_action(self, g) -> int:
        _, best_action = self.minimax(g, self.max_depth, float('-inf'), float('inf'), True)
        return best_action if best_action is not None else 0



class MyMinimaxWithAlphaBetaSortedKiller(BattlePolicy):
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.name = "Minimax with pruning alpha beta sorted killer"
        self.killer_moves = {depth: [] for depth in range(max_depth + 1)}  # Killer moves per profondità

    def minimax(self, g, depth, alpha, beta, is_maximizing_player):
        if depth == 0:
            return game_state_eval(g, depth), None

        if is_maximizing_player:
            max_eval = float('-inf')
            best_action = None

            # Ottieni le mosse disponibili
            moves = list(range(DEFAULT_N_ACTIONS))

            # Ordina le mosse: killer moves prima, poi in base alla valutazione preliminare
            moves = self._sort_moves(g, moves, depth, is_maximizing_player)

            for i in moves:
                g_copy = deepcopy(g)
                s, _, _, _, _ = g_copy.step([i, 99])
                if n_fainted(s[0].teams[0]) > n_fainted(g.teams[0]):
                    continue

                eval_score, _ = self.minimax(s[0], depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = i

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    # Aggiorna killer moves
                    if i not in self.killer_moves[depth]:
                        self.killer_moves[depth].append(i)
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth].pop(0)
                    break
            return max_eval, best_action

        else:
            min_eval = float('inf')
            best_action = None

            # Ottieni le mosse disponibili
            moves = list(range(DEFAULT_N_ACTIONS))

            # Ordina le mosse: killer moves prima, poi in base alla valutazione preliminare
            moves = self._sort_moves(g, moves, depth, is_maximizing_player)

            for j in moves:
                g_copy = deepcopy(g)
                s, _, _, _, _ = g_copy.step([99, j])
                if n_fainted(s[0].teams[1]) > n_fainted(g.teams[1]):
                    continue

                eval_score, _ = self.minimax(s[0], depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = j

                beta = min(beta, eval_score)
                if beta <= alpha:
                    # Aggiorna killer moves
                    if j not in self.killer_moves[depth]:
                        self.killer_moves[depth].append(j)
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth].pop(0)
                    break
            return min_eval, best_action

    def _sort_moves(self, g, moves, depth, is_maximizing_player):
        """
        Ordina le mosse basandosi sulle killer moves e sulla valutazione preliminare.

        :param g: Lo stato corrente del gioco.
        :param moves: Lista di mosse disponibili.
        :param depth: Profondità corrente.
        :param is_maximizing_player: Se il giocatore corrente è massimizzante.
        :return: Lista di mosse ordinate.
        """
        # Valuta ogni mossa
        move_scores = []
        for move in moves:
            g_copy = deepcopy(g)
            s, _, _, _, _ = g_copy.step([move, 99] if is_maximizing_player else [99, move])
            score = game_state_eval(s[0], depth)
            move_scores.append((move, score))

        # Ordina: killer moves prima, poi in base al punteggio
        killer_moves = self.killer_moves.get(depth, [])
        move_scores.sort(key=lambda x: (x[0] not in killer_moves, -x[1] if is_maximizing_player else x[1]))
        return [move for move, _ in move_scores]

    def get_action(self, g) -> int:
        _, best_action = self.minimax(g, self.max_depth, float('-inf'), float('inf'), True)
        return best_action if best_action is not None else 0


class MyMonteCarloWithMinimax(BattlePolicy):
    def __init__(self, max_iterations: int = 200, exploration_weight: float = 1.8, minimax_depth: int = 5, dynamic_minimax: bool = True):
        """
        Inizializza la politica ibrida MCTS + Minimax con valutazione dinamica.

        :param max_iterations: Numero massimo di iterazioni per MCTS.
        :param exploration_weight: Peso per il termine di esplorazione (valore di C).
        :param minimax_depth: Profondità massima per Minimax (default).
        :param dynamic_minimax: Abilita/disabilita la valutazione dinamica della profondità di Minimax.
        """
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight
        self.minimax_depth = minimax_depth
        self.dynamic_minimax = dynamic_minimax
        self.name = "Monte Carlo with Minimax"

    def get_dynamic_minimax_depth(self, num_promising_nodes):
        """
        Determina la profondità dinamica di Minimax in base al numero di nodi promettenti.

        :param num_promising_nodes: Numero di nodi promettenti da analizzare.
        :return: Profondità di Minimax regolata dinamicamente.
        """
        if not self.dynamic_minimax:
            return self.minimax_depth

        base_depth = self.minimax_depth
        if num_promising_nodes <= 2:
            return base_depth + 2  # Più profondità se pochi nodi
        elif num_promising_nodes <= 5:
            return base_depth + 1  # Lieve aumento di profondità
        else:
            return base_depth  # Usa profondità predefinita

    def mcts(self, g, root_player: int):
        """
        Implementa il processo di Monte Carlo Tree Search (MCTS).
        """
        root = Node(state=deepcopy(g), parent=None, player=root_player)

        for _ in range(self.max_iterations):
            leaf = self._select(root)
            simulation_result = self._simulate(leaf.state, leaf.player)
            self._backpropagate(leaf, simulation_result)

        return root

    def _select(self, node):
        """
        Seleziona un nodo utilizzando UCT (Upper Confidence Bound for Trees).
        """
        while node.children:
            node = max(node.children, key=lambda child: self._uct(child))
        if not node.is_fully_expanded():
            return self._expand(node)
        return node

    def _expand(self, node):
        """
        Espande il nodo generando un nuovo figlio.
        """
        untried_actions = node.get_untried_actions()
        action = random.choice(untried_actions)
        new_state, _, _, _, _ = deepcopy(node.state).step([action, 99])
        child_node = Node(state=new_state, parent=node, player=1 - node.player, action=action)
        node.children.append(child_node)
        return child_node

    def _simulate(self, state, player):
        """
        Esegue una simulazione usando una politica greedy a partire da uno stato.
        
        :param state: Lo stato iniziale della simulazione.
        :param player: Il giocatore corrente (0 o 1).
        :return: Il risultato della simulazione.
        """
        g_copy = deepcopy(state)
        current_player = player

        while not g_copy.is_terminal():
            # Usa la politica greedy per scegliere l'azione migliore
            best_action = self._greedy_action(g_copy, current_player)
            g_copy.step([best_action, 99] if current_player == 0 else [99, best_action])

            # Cambia giocatore
            current_player = 1 - current_player

        # Valuta lo stato finale per il giocatore iniziale
        return self._evaluate(g_copy, player)

    def _greedy_action(self, state, player):
        """
        Sceglie la migliore azione per il giocatore corrente utilizzando una politica greedy.
        
        :param state: Lo stato corrente.
        :param player: Il giocatore che effettua la mossa (0 o 1).
        :return: L'azione migliore da eseguire.
        """
        best_action = None
        best_value = float('-inf') if player == 0 else float('inf')  # Massimizza per il player 0, minimizza per il player 1

        for action in range(DEFAULT_N_ACTIONS):
            g_copy = deepcopy(state)
            new_state, _, _, _, _ = g_copy.step([action, 99] if player == 0 else [99, action])

            # Valuta lo stato risultante
            eval_value = self._evaluate(new_state, player)

            # Aggiorna l'azione migliore
            if (player == 0 and eval_value > best_value) or (player == 1 and eval_value < best_value):
                best_value = eval_value
                best_action = action

        return best_action if best_action is not None else random.randint(0, DEFAULT_N_ACTIONS - 1)

    def _evaluate(self, state, player):
        """
        Valuta lo stato finale per un giocatore specifico.
        """
        return game_state_eval(state, player)

    def _backpropagate(self, node, result):
        """
        Propaga i risultati della simulazione verso la radice.
        """
        while node:
            node.visits += 1
            node.value += result if node.player == node.parent.player else -result
            node = node.parent

    def _uct(self, node):
        """
        Calcola l'Upper Confidence Bound per un nodo.
        """
        if node.visits == 0:
            return float('inf')  # Esplora nodi non visitati
        return (node.value / node.visits +
                self.exploration_weight * math.sqrt(math.log(node.parent.visits) / node.visits))

    def get_action(self, g) -> int:
        """
        Trova la migliore azione combinando MCTS e Minimax.
        """
        # Fase 1: Usa MCTS per esplorare i primi livelli
        root = self.mcts(g, root_player=0)

        # Fase 2: Seleziona i nodi più promettenti in base alle visite e ai valori
        promising_nodes = sorted(root.children, key=lambda child: (child.visits, child.value), reverse=True)[:3]

        # Determina la profondità dinamica di Minimax
        dynamic_depth = self.get_dynamic_minimax_depth(len(promising_nodes))

        # Fase 3: Usa Minimax per approfondire sui nodi selezionati
        best_action = None
        best_value = float('-inf')

        for node in promising_nodes:
            eval_score, _ = self.minimax(node.state, dynamic_depth, float('-inf'), float('inf'), True)
            if eval_score > best_value:
                best_value = eval_score
                best_action = node.action

        return best_action if best_action is not None else 0 

    def minimax(self, g, depth, alpha, beta, is_maximizing_player):
        """
        Algoritmo Minimax con Alpha-Beta Pruning.
        """
        if depth == 0:
            return game_state_eval(g, depth), None

        if is_maximizing_player:
            max_eval = float('-inf')
            best_action = None
            for i in range(DEFAULT_N_ACTIONS):
                g_copy = deepcopy(g)
                s, _, _, _, _ = g_copy.step([i, 99])
                if n_fainted(s[0].teams[0]) > n_fainted(g.teams[0]):
                    continue
                eval_score, _ = self.minimax(s[0], depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = i
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_action

        else:
            min_eval = float('inf')
            best_action = None
            for j in range(DEFAULT_N_ACTIONS):
                g_copy = deepcopy(g)
                s, _, _, _, _ = g_copy.step([99, j])
                if n_fainted(s[0].teams[1]) > n_fainted(g.teams[1]):
                    continue
                eval_score, _ = self.minimax(s[0], depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = j
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_action

class MCTS_MR(BattlePolicy):
    def __init__(self, max_iterations: int = 1000, exploration_weight: float = 1.41, minimax_depth: int = 2):
        """
        Inizializza la politica Monte Carlo Tree Search con Minimax Rollouts (MCTS-MR).

        :param max_iterations: Numero massimo di iterazioni per MCTS.
        :param exploration_weight: Peso per il termine di esplorazione (valore di C).
        :param minimax_depth: Profondità massima per le ricerche Minimax durante i rollout.
        """
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight
        self.minimax_depth = minimax_depth
        self.name = "Monte Carlo with Minimax Rollouts"

    def mcts(self, g, root_player: int):
        # Nodo radice
        root = Node(state=deepcopy(g), parent=None, player=root_player)

        for _ in range(self.max_iterations):
            leaf = self._select(root)
            simulation_result = self._simulate(leaf.state, leaf.player)
            self._backpropagate(leaf, simulation_result)

        # Scegli la migliore azione dalla radice
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.action

    def _simulate(self, state, player):
        """
        Esegue una simulazione informata a partire da uno stato, usando Minimax quando possibile.

        :param state: Lo stato iniziale della simulazione.
        :param player: Il giocatore corrente.
        :return: Il risultato della simulazione.
        """
        g_copy = deepcopy(state)
        while not g_copy.is_terminal():
            # Usa minimax per scegliere la prossima azione nei rollout
            action = self._minimax_action(g_copy, self.minimax_depth, player)
            g_copy.step([action, 99])  # L'avversario esegue un'azione non valida
        return self._evaluate(g_copy, player)

    def _minimax_action(self, g, depth, player):
        """
        Usa una ricerca Minimax a profondità limitata per scegliere un'azione.

        :param g: Lo stato corrente del gioco.
        :param depth: La profondità massima della ricerca.
        :param player: Il giocatore corrente (0 o 1).
        :return: L'azione migliore trovata tramite Minimax.
        """
        def minimax(g, depth, is_maximizing):
            if depth == 0 or g.is_terminal():
                return game_state_eval(g, player), None

            if is_maximizing:
                max_eval = float('-inf')
                best_action = None
                for i in range(DEFAULT_N_ACTIONS):
                    g_copy = deepcopy(g)
                    s, _, _, _, _ = g_copy.step([i, 99])
                    eval_score, _ = minimax(s[0], depth - 1, False)
                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_action = i
                return max_eval, best_action
            else:
                min_eval = float('inf')
                best_action = None
                for j in range(DEFAULT_N_ACTIONS):
                    g_copy = deepcopy(g)
                    s, _, _, _, _ = g_copy.step([99, j])
                    eval_score, _ = minimax(s[0], depth - 1, True)
                    if eval_score < min_eval:
                        min_eval = eval_score
                        best_action = j
                return min_eval, best_action

        _, best_action = minimax(g, depth, True)
        return best_action if best_action is not None else random.randint(0, DEFAULT_N_ACTIONS - 1)

    def _evaluate(self, state, player):
        return game_state_eval(state, player)

    def _select(self, node):
        while node.children:
            node = max(node.children, key=lambda child: self._uct(child))
        if not node.is_fully_expanded():
            return self._expand(node)
        return node

    def _expand(self, node):
        untried_actions = node.get_untried_actions()
        action = random.choice(untried_actions)
        new_state, _, _, _, _ = deepcopy(node.state).step([action, 99])
        child_node = Node(state=new_state, parent=node, player=1 - node.player, action=action)
        node.children.append(child_node)
        return child_node

    def _backpropagate(self, node, result):
        while node:
            node.visits += 1
            node.value += result if node.player == node.parent.player else -result
            node = node.parent

    def _uct(self, node):
        if node.visits == 0:
            return float('inf')
        return (node.value / node.visits +
                self.exploration_weight * math.sqrt(math.log(node.parent.visits) / node.visits))

    def get_action(self, g) -> int:
        return self.mcts(g, root_player=0)


class MyMinimaxWithAlphaBetaKillertransposition(BattlePolicy):
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.name = "Minimax with pruning alpha beta killer transposition"
        self.killer_moves = {depth: [] for depth in range(max_depth + 1)}  # Killer moves per depth
        self.transposition_table = {}  # Transposition table

    def _hash_state(self, g):
        """
        Hash the game state to use as a key in the transposition table.
        """
        return hashlib.md5(str(g).encode()).hexdigest()

    def minimax(self, g, depth, alpha, beta, is_maximizing_player):
        state_hash = self._hash_state(g)

        # Check the transposition table
        if state_hash in self.transposition_table:
            entry = self.transposition_table[state_hash]
            if entry['depth'] >= depth:
                if entry['flag'] == 'EXACT':
                    return entry['value'], entry['best_action']
                elif entry['flag'] == 'LOWERBOUND' and entry['value'] > alpha:
                    alpha = entry['value']
                elif entry['flag'] == 'UPPERBOUND' and entry['value'] < beta:
                    beta = entry['value']
                if alpha >= beta:
                    return entry['value'], entry['best_action']

        # Terminal condition or depth limit
        if depth == 0:
            eval_score = game_state_eval(g, depth)
            self.transposition_table[state_hash] = {
                'value': eval_score,
                'depth': depth,
                'flag': 'EXACT',
                'best_action': None
            }
            return eval_score, None

        best_action = None

        if is_maximizing_player:
            max_eval = float('-inf')
            moves = list(range(DEFAULT_N_ACTIONS))
            moves = self._sort_moves(g, moves, depth, is_maximizing_player)

            for i in moves:
                g_copy = deepcopy(g)
                s, _, _, _, _ = g_copy.step([i, 99])
                if n_fainted(s[0].teams[0]) > n_fainted(g.teams[0]):
                    continue

                eval_score, _ = self.minimax(s[0], depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = i

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    # Update killer moves
                    if i not in self.killer_moves[depth]:
                        self.killer_moves[depth].append(i)
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth].pop(0)
                    break

            # Store in the transposition table
            self.transposition_table[state_hash] = {
                'value': max_eval,
                'depth': depth,
                'flag': 'EXACT' if alpha < beta else 'LOWERBOUND',
                'best_action': best_action
            }
            return max_eval, best_action

        else:
            min_eval = float('inf')
            moves = list(range(DEFAULT_N_ACTIONS))
            moves = self._sort_moves(g, moves, depth, is_maximizing_player)

            for j in moves:
                g_copy = deepcopy(g)
                s, _, _, _, _ = g_copy.step([99, j])
                if n_fainted(s[0].teams[1]) > n_fainted(g.teams[1]):
                    continue

                eval_score, _ = self.minimax(s[0], depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = j

                beta = min(beta, eval_score)
                if beta <= alpha:
                    # Update killer moves
                    if j not in self.killer_moves[depth]:
                        self.killer_moves[depth].append(j)
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth].pop(0)
                    break

            # Store in the transposition table
            self.transposition_table[state_hash] = {
                'value': min_eval,
                'depth': depth,
                'flag': 'EXACT' if alpha < beta else 'UPPERBOUND',
                'best_action': best_action
            }
            return min_eval, best_action

    def _sort_moves(self, g, moves, depth, is_maximizing_player):
        """
        Sort moves based on killer moves and preliminary evaluation.
        """
        move_scores = []
        for move in moves:
            g_copy = deepcopy(g)
            s, _, _, _, _ = g_copy.step([move, 99] if is_maximizing_player else [99, move])
            score = game_state_eval(s[0], depth)
            move_scores.append((move, score))

        # Sort: killer moves first, then by evaluation score
        killer_moves = self.killer_moves.get(depth, [])
        move_scores.sort(key=lambda x: (x[0] not in killer_moves, -x[1] if is_maximizing_player else x[1]))
        return [move for move, _ in move_scores]

    def get_action(self, g) -> int:
        _, best_action = self.minimax(g, self.max_depth, float('-inf'), float('inf'), True)
        return best_action if best_action is not None else 0
