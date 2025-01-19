import tkinter
from copy import deepcopy
from threading import Thread, Event
from tkinter import CENTER, DISABLED, NORMAL
from types import CellType
from typing import List
from random import random
import numpy as np
from customtkinter import CTk, CTkButton, CTkRadioButton, CTkLabel
import pickle
from vgc.behaviour import BattlePolicy,BattlePolicies
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER, DEFAULT_N_ACTIONS, MAX_HIT_POINTS, STATE_DAMAGE, SPIKES_2, SPIKES_3
from vgc.datatypes.Objects import GameState, PkmTeam,PkmMove, Pkm
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition, PkmStatus, PkmEntryHazard, MAX_STAGE, MIN_STAGE
import math
import hashlib
from typing import List
import numpy as np
from vgc.competition.StandardPkmMoves import Struggle
from operator import itemgetter

from typing import Tuple, List, Dict
import heapq
import copy
import sys 
import termios

import numpy as np
from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE
from vgc.datatypes.Objects import GameState
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition


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
        self.name = "Random Player"

    def get_action(self, g: GameState) -> int:
        return np.random.choice(self.n_actions, p=self.pi)




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


#provare a fare qualcosa con la forza di un pokemon magari la forza delle sue mosse
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


def aggressive_eval(state):
    """
    Valutazione dello stato di gioco per un agente aggressivo.
    """
    mine = state.teams[0].active
    opp = state.teams[1].active

    # Valuta il massimo danno possibile con la mossa più potente super-efficac
    
    efficacia = mine.type.get_super_effective(opp.type)
    

    for move in mine.moves:
        # i check if my moves still have some pp
        if move.pp > 0:  
            if move.type == mine.type :
                stab = 1.5
            

    # Penalità per la salute residua dell'avversario
    opp_hp_penalty = opp.hp / opp.max_hp

    # Combina i fattori
    score = max_effective_damage - opp_hp_penalty
    return score



def My_game_state_eval(state, depth):
    """
    Funzione di valutazione avanzata basata sull'impatto delle mosse più potenti.
    """
    # i get my pokemon from the roster
    mine = state.teams[0].active
    opp = state.teams[1].active

    # Calcola il massimo danno che il Pokémon attivo può infliggere
    max_damage_to_opponent = max(estimate_damage(move.type, mine.type,move.power,opp.type,mine.status,opp.status,move.weather) for move in mine.moves)

    # Calcola il massimo danno che il Pokémon avversario può infliggere
    #max_damage_from_opponent = max(estimate_damage(move.type, opp.type,move.power,mine.type,opp.status,mine.status,move.weather) for move in opp.moves)

    # Valuta la differenza in termini di potenziale offensivo e difensivo
    #damage_balance = max_damage_to_opponent - max_damage_from_opponent

    # Considera la vita residua
    hp_balance = mine.hp / mine.max_hp - opp.hp / opp.max_hp
    

    # Considera il numero di Pokémon rimanenti per entrambe le squadre
    #team_balance = len([p for p in state.teams[0].pokemons if not p.fainted]) - len([p for p in state.teams[1].pokemons if not p.fainted])

    # Combina i fattori in un punteggio
    score = 2 * damage_balance + hp_balance + 0.5 * team_balance - 0.05 * depth
    return score




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
        self.name = 'Minimax'

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
        self.name = "TunedTreeTraversal"
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
            return aggressive_eval(g, depth), None

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
                        if len(self.killer_moves[depth]) > 2: 
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

            # per ottenere le mosse disponibili
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



class MyMinimaxWithAlphaBetaKillertransposition(BattlePolicy):
    def __init__(self, max_depth: int = 12):
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

class MCTS_MS(BattlePolicy):
    def __init__(self, max_iterations: int = 1000, exploration_weight: float = 1.41, minimax_depth: int = 2):
        """
        Inizializza la politica Monte Carlo Tree Search con Minimax Selection (MCTS-MS).

        :param max_iterations: Numero massimo di iterazioni per MCTS.
        :param exploration_weight: Peso per il termine di esplorazione (valore di C).
        :param minimax_depth: Profondità massima per le ricerche Minimax durante la selezione.
        """
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight
        self.minimax_depth = minimax_depth
        self.name = "Monte Carlo with Minimax Selection"

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

    def _select(self, node):
        """
        Seleziona un nodo utilizzando UCT e Minimax Selection.

        :param node: Il nodo corrente.
        :return: Il nodo foglia selezionato.
        """
        while node.children:
            # Applica minimax se un nodo raggiunge un numero minimo di visite
            if node.visits >= 5:  
                minimax_result = self._minimax_action(node.state, self.minimax_depth, node.player)
                if minimax_result is not None:
                    return Node(state=deepcopy(node.state), parent=node, player=node.player)

            # Se non soddisfa i criteri di minimax, usa UCT per scegliere
            node = max(node.children, key=lambda child: self._uct(child))
        
        # continua ad espandere il nodo
        if not node.is_fully_expanded():
            return self._expand(node)
        return node

    def _minimax_action(self, g, depth, player):
        """
        Usa una ricerca Minimax a profondità limitata per valutare uno stato.

        :param g: Lo stato corrente del gioco.
        :param depth: La profondità massima della ricerca.
        :param player: Il giocatore corrente (0 o 1).
        :return: Una valutazione Minimax dello stato (se applicabile), altrimenti None.
        """
        def minimax(g, depth, is_maximizing):
            if depth == 0 or g.is_terminal():
                return game_state_eval(g, player)

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

    def _simulate(self, state, player):
        g_copy = deepcopy(state)
        while not g_copy.is_terminal():
            action = random.randint(0, DEFAULT_N_ACTIONS - 1)
            g_copy.step([action, 99]) 
        return self._evaluate(g_copy, player)

    def _evaluate(self, state, player):
        return game_state_eval(state, player)

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


























# Eval functions
def estimate_damage(move_type: PkmType, pkm_type: PkmType, move_power: float, opp_pkm_type: PkmType,
                    attack_stage: int, defense_stage: int, weather: WeatherCondition) -> float:
    stab = 1.5 if move_type == pkm_type else 1.
    # condizione favorevole 
    if (move_type == PkmType.WATER and weather == WeatherCondition.RAIN) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.SUNNY):
        weather = 1.5
    # condizione sfavorevole
    elif (move_type == PkmType.WATER and weather == WeatherCondition.SUNNY) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.RAIN):
        weather = .5
    else:
        weather = 1.
    
    stage_level = attack_stage - defense_stage
    # da vedere 
    stage = (stage_level + 2.) / 2 if stage_level >= 0. else 2. / (np.abs(stage_level) + 2.)
    # stima del danno
    damage = TYPE_CHART_MULTIPLIER[move_type][opp_pkm_type] * stab * weather * stage * move_power
    return damage


def evaluate_matchup(pkm_type: PkmType, opp_pkm_type: PkmType, moves_type: List[PkmType]) -> float:
    for mtype in moves_type: # cerca mossa super efficace 
        if TYPE_CHART_MULTIPLIER[mtype][pkm_type] == 2.0:
            return 2.0  # ritorna 2 nel caso in cui ci sia una mossa super efficace 
    # altrimenti considera solo la valutazione rispetto al tipo di pokemon 
    return TYPE_CHART_MULTIPLIER[opp_pkm_type][pkm_type]


# My Battle Policy
class MyPolicy(BattlePolicy):

    def __init__(self):
        self.hail_used = False
        self.sandstorm_used = False
        self.name = "My Policy"


    def get_action(self, g: GameState) -> int:

        # la mia squadra
        my_team = g.teams[0]
        active_pkm = my_team.active
        bench = my_team.party
        my_attack_stage = my_team.stage[PkmStat.ATTACK]

        # squadra avversaria
        opp_team = g.teams[1]
        opp_active_pkm = opp_team.active
        opp_not_fainted_pkms = len(opp_team.get_not_fainted())
        opp_attack_stage = opp_team.stage[PkmStat.ATTACK]
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        # meteo 
        weather = g.weather.condition
        
        # valutazione mosse
        damage: List[float] = []
        for move in active_pkm.moves:
            damage.append(estimate_damage(move.type, active_pkm.type, move.power, opp_active_pkm.type, my_attack_stage,
                                          opp_defense_stage, weather))
        # scegli mossa
        move_id = int(np.argmax(damage))
        # se elimina l'avversario oppure il tipo di mossa è superefficace si usa subito: 
        if (damage[move_id] >= opp_active_pkm.hp) or (damage[move_id] > 0 and TYPE_CHART_MULTIPLIER[active_pkm.moves[move_id].type][opp_active_pkm.type] == 2.0) :
            return move_id
        
        defense_type_multiplier = evaluate_matchup(active_pkm.type, opp_active_pkm.type,
                                                   list(map(lambda m: m.type, opp_active_pkm.moves)))
        if defense_type_multiplier <= 1.0: 
            return move_id
        # if defense_type_multiplier <= 1.0:
            # # se l'avversario non ha ancora messo le punte e la battaglia è ancora lunga 
            # if opp_team.entry_hazard != PkmEntryHazard.SPIKES and opp_not_fainted_pkms >= 2 and opp_active_pkm.hp == opp_active_pkm.max_hp:
            #     for i in range(DEFAULT_PKM_N_MOVES):  
            #         if active_pkm.moves[i].hazard == PkmEntryHazard.SPIKES: 
            #             print("spine")
            #             input()
            #             return i # metto le spine, se le ho
                    
            # # stessa cosa per la tempesta di sabbia
            # if weather != WeatherCondition.SANDSTORM and not self.sandstorm_used and defense_type_multiplier < 1.0:
            #     sandstorm_move = -1
            #     for i in range(DEFAULT_PKM_N_MOVES):
            #         if active_pkm.moves[i].weather == WeatherCondition.SANDSTORM:
            #             sandstorm_move = i
            #     immune_pkms = 0
            #     for pkm in bench:
            #         if not pkm.fainted() and pkm.type in [PkmType.GROUND, PkmType.STEEL, PkmType.ROCK]:
            #             immune_pkms += 1
            #     # solo se i miei pokemon sono tutti e tre immuni alla sabbia
            #     if sandstorm_move != -1 and immune_pkms >= 2 and opp_not_fainted_pkms >= 2: 
            #         self.sandstorm_used = True
            #         return sandstorm_move

            # if weather != WeatherCondition.HAIL and not self.hail_used and defense_type_multiplier < 1.0:
            #     hail_move = -1
            #     for i in range(DEFAULT_PKM_N_MOVES):
            #         if active_pkm.moves[i].weather == WeatherCondition.HAIL:
            #             hail_move = i
            #     immune_pkms = 0
            #     for pkm in bench:
            #         if not pkm.fainted() and pkm.type in [PkmType.ICE]:
            #             immune_pkms += 1
            #     if hail_move != -1 and immune_pkms == 3:
            #         self.hail_used = True
            #         return hail_move
            # return move_id
        
 
        # considera il cambio pokemon
        matchup: List[float] = []
        not_fainted = False
        active_idx = 0
        for j in range(len(bench)):
            if bench[j] == active_pkm: 
                active_idx = j
            if bench[j].hp == 0.0:
                matchup.append(0.0)
            else:
                not_fainted = True
                matchup.append(
                    evaluate_matchup(bench[j].type, opp_active_pkm.type, list(map(lambda m: m.type, opp_active_pkm.moves))))
    
        best_switch = int(np.argmin(matchup))
        if not_fainted and bench[best_switch] != active_pkm and (
                evaluate_matchup(bench[best_switch].type, opp_active_pkm.type,list(map(lambda m: m.type, opp_active_pkm.moves)) >= (bench[active_idx].type, opp_active_pkm.type,list(map(lambda m: m.type, opp_active_pkm.moves)))+1)) :
            print(matchup)
            input()
            return best_switch + 4
        
        return move_id


def aggressive_eval(state):
    """
    Valutazione basata su:
    - Danno massimo con mosse super efficaci.
    - Bonus STAB.
    - Velocità per attaccare per primi.
    - Penalità per lo stato e la salute residua.
    """
    mine = state.teams[0].active
    opp = state.teams[1].active

    best_damage = 0
    for move in mine.moves:
        if move.pp > 0 and not move.has_negative_effect_on_user():  # Ignora mosse esaurite o con effetti negativi
            damage = calculate_damage(move, mine, opp)
            if mine.speed > opp.speed:  # Bonus velocità
                damage *= 1.1
            best_damage = max(best_damage, damage)

    # Penalità se il Pokémon avversario è super efficace
    if opp.is_super_effective_against(mine):
        return -float('inf')  # Forza uno switch

    return best_damage

def choose_switch(state):
    """
    Sceglie un Pokémon da mandare in campo in base al tipo del Pokémon avversario.
    """
    mine = state.teams[0]
    opp = state.teams[1].active

    best_switch = None
    best_score = -float('inf')

    for pokemon in mine.all_pokemon:
        if not pokemon.is_fainted() and pokemon != mine.active:
            effectiveness = pokemon.effectiveness_against(opp.types)
            if effectiveness > best_score:
                best_score = effectiveness
                best_switch = pokemon

    return best_switch

def calculate_damage(move, attacker, defender):
    """
    Calcola il danno di una mossa tenendo conto di efficacia e STAB.
    """
    effectiveness = move.effectiveness_against(defender.types)
    stab = 1.5 if move.type in attacker.types else 1.0
    base_damage = move.power * attacker.attack / defender.defense
    return base_damage * effectiveness * stab



class AggressiveMinimaxWithMonteCarlo(BattlePolicy):
    def __init__(self, max_depth: int = 20, n_simulations: int = 30):
        self.max_depth = max_depth
        self.n_simulations = n_simulations
        self.name = "Aggressive Minimax with Monte Carlo and Killer Moves"
        self.killer_moves = {depth: [] for depth in range(max_depth + 1)}  # Killer moves per profondità

    def monte_carlo_evaluation(self, g, move, is_maximizing_player):
        """
        Esegue simulazioni Monte Carlo per valutare l'efficacia di una mossa.
        """
        total_score = 0
        for _ in range(self.n_simulations):
            g_copy = deepcopy(g)
            if is_maximizing_player:
                g_copy.step([move, random.randint(0, DEFAULT_N_ACTIONS - 1)])
            else:
                g_copy.step([random.randint(0, DEFAULT_N_ACTIONS - 1), move])
            total_score += self.simulate_random_game(g_copy, is_maximizing_player)
        return total_score / self.n_simulations

    def simulate_random_game(self, g, is_maximizing_player):
        """
        Simula una partita casuale fino alla fine e restituisce una valutazione dello stato.
        """
        g_copy = deepcopy(g)
        while not g_copy.is_terminal():
            g_copy.step([
                random.randint(0, DEFAULT_N_ACTIONS - 1),
                random.randint(0, DEFAULT_N_ACTIONS - 1)
            ])
        return game_state_eval(g_copy, 0)

    

    def best_switch(self, my_team, opponent):
        """
        Trova il miglior Pokémon da far entrare in base al tipo dell'avversario.
        """
        best_switch_score = float('-inf')
        best_switch = None
        for pokemon in my_team:
            if not pokemon.is_fainted:
                score = TYPE_CHART_MULTIPLIER[pokemon.primary_type][opponent.type]
                if score > best_switch_score:
                    best_switch_score = score
                    best_switch = pokemon
        return best_switch

    def minimax(self, g, depth, alpha, beta, is_maximizing_player):
        if depth == 0 or g.is_terminal():
            return game_state_eval(g, depth), None

        if is_maximizing_player:
            max_eval = float('-inf')
            best_action = None
            moves = list(range(DEFAULT_N_ACTIONS))
            killer_moves = self.killer_moves.get(depth, [])
            moves = sorted(moves, key=lambda move: move in killer_moves, reverse=True)

            for action in moves:
                g_copy = deepcopy(g)
                s, _, _, _, _ = g_copy.step([action, 99])
                if n_fainted(s[0].teams[0]) > n_fainted(g.teams[0]):
                    continue

                # Usa Monte Carlo per valutare la mossa
                eval_score = self.monte_carlo_evaluation(s[0], action, True)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    if action not in self.killer_moves[depth]:
                        self.killer_moves[depth].append(action)
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth].pop(0)
                    break
            return max_eval, best_action

        else:
            min_eval = float('inf')
            best_action = None
            moves = list(range(DEFAULT_N_ACTIONS))
            killer_moves = self.killer_moves.get(depth, [])
            moves = sorted(moves, key=lambda move: move in killer_moves, reverse=True)

            for action in moves:
                g_copy = deepcopy(g)
                s, _, _, _, _ = g_copy.step([99, action])
                if n_fainted(s[0].teams[1]) > n_fainted(g.teams[1]):
                    continue

                # Usa Monte Carlo per valutare la mossa
                eval_score = self.monte_carlo_evaluation(s[0], action, False)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action

                beta = min(beta, eval_score)
                if beta <= alpha:
                    if action not in self.killer_moves[depth]:
                        self.killer_moves[depth].append(action)
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth].pop(0)
                    break
            return min_eval, best_action


    def evaluate_switch(self, active_pokemon, opponent_pokemon, bench):
        """
        Valuta se effettuare uno switch e quale Pokémon scegliere.
        """
        best_switch = None
        best_score = float('-inf')

        for bench_pokemon in bench:
            if bench_pokemon.is_fainted:  # Salta Pokémon esausti
                continue
            
            # Calcola vulnerabilità e vantaggio contro l'avversario
            vulnerability = TYPE_CHART_MULTIPLIER[opponent_pokemon.primary_type][bench_pokemon.primary_type]
            effectiveness = TYPE_CHART_MULTIPLIER[bench_pokemon.primary_type][opponent_pokemon.primary_type]

            # Valuta il punteggio: più è efficace contro l'avversario e meno è vulnerabile, meglio è
            score = effectiveness - vulnerability

            if score > best_score:
                best_score = score
                best_switch = bench_pokemon

        # Effettua lo switch solo se c'è un'opzione migliore rispetto all'attivo
        current_vulnerability = TYPE_CHART_MULTIPLIER[opponent_pokemon.primary_type][active_pokemon.primary_type]

        if best_switch and (best_score > 1.0 or current_vulnerability > 1.0):
            return best_switch

        return None


    def evaluate_status_and_weather(self, active_pokemon, opponent_pokemon, weather):
        """
        Valuta lo stato e il meteo per decidere se fare uno switch.
        """
        # Valuta se il meteo è sfavorevole
        if weather:
            if weather.is_unfavorable_for(active_pokemon) and not weather.is_unfavorable_for(opponent_pokemon):
                return True  # Suggerisce uno switch

        # Valuta se lo status del Pokémon avversario può essere sfruttato
        if opponent_pokemon.status in [5, 4, 3, 1]:  # Paralizzato, addormentato, congelato e confuso
            return False  # Non switchare, puoi sfruttare lo status avversario

        return False  # Non cambiare in condizioni neutre


    def get_action(self, g):
        """
        Calcola la migliore azione basata su minimax, con considerazioni avanzate per switch e attacchi.
        """
        active_pokemon = g.teams[0].active
        opponent_pokemon = g.teams[1].active
        bench = g.teams[0].team[1:]  # Pokémon in panchina
        weather = g.weather

        # 1. Valuta se fare uno switch
        if self.evaluate_status_and_weather(active_pokemon, opponent_pokemon, weather):
            best_switch = self.evaluate_switch(active_pokemon, opponent_pokemon, bench)
            if best_switch:
                return f"switch to {best_switch.name}"

        # 2. Trova la migliore mossa
        best_move = None
        best_score = float('-inf')

        for move in active_pokemon.moves:
            if move:
                score = self.evaluate_move(move, active_pokemon, opponent_pokemon)
                if score > best_score:
                    best_score = score
                    best_move = move

        if best_move:
            return best_move.index

        # 3. in caso non trovi nessuna mossa che possa andare bene allora va prendere una mossa random
        return random.randint(0, DEFAULT_N_ACTIONS - 1)







class AggressivePolicy(BattlePolicy):
    def __init__(self, switch_probability: float = .15, n_moves: int = DEFAULT_PKM_N_MOVES,
                 n_switches: int = DEFAULT_PARTY_SIZE):
        # super.init()

        self.hail_used = False 
        self.sandstorm_used = False
        self.name = "Aggressive Policy"

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass


    # def evaluate_move(self, move, active_pokemon, opponent_pokemon):
    #     """
    #     Valuta una mossa considerando la STAB, l'efficacia e la velocità.
    #     """
    #     effectiveness = move.type.effectiveness(opponent_pokemon.type)
    #     stab = 1.5 if move.type in active_pokemon.type else 1.0
    #     speed_priority = 1 if active_pokemon.speed > opponent_pokemon.speed else -1
    #     return move.power * effectiveness * stab + speed_priority * 10

    # def evaluate_switch(self, active_pokemon, opponent_pokemon, bench):
    #     """
    #     Valuta se effettuare uno switch e quale Pokémon usare.
    #     """
    #     matchup: List[float] = []
    #     not_fainted = False
    #     for bench_pokemon in bench:
    #         if bench_pokemon.hp == 0.0: 
    #             matchup.append(0.0)
    #         else: 
    #             not_fainted = True
    #             matchup.append(
    #                 evaluate_matchup(bench_pokemon.type, opponent_pokemon.type, list(map(lambda m: m.type, opponent_pokemon.moves))))
    #     best_switch = int(np.argmin(matchup))
    #     if not_fainted and bench_pokemon[best_switch] != active_pokemon:
    #         print("Switching to someone else")
    #         input()
    #         return best_switch + 4

    def get_action(self, state):
        
        """
        Restituisce la mossa o lo switch migliore.
        """
        # tempo 
        weather = state.weather.condition
        # la mia squadra
        active_pokemon = state.teams[0].active #pokemon in campo
        bench = state.teams[0].party # panchina 
        my_attack_stage = state[0].stage[PkmStat.ATTACK]
        my_defense_stage = state[0].stage[PkmStat.DEFENSE]

        # squadra avversaria 
        opponent_pokemon = state.teams[1].active # pkm avversario in campo
        opp_attack_stage = state[1].stage[PkmStat.ATTACK]
        opp_defense_stage = state[1].stage[PkmStat.DEFENSE]
        opp_not_fainted_pkms = len(state[1].get_not_fainted())

        # Valuta le mosse disponibili
        damage: List[float] = []
        for move in active_pokemon.moves:
            damage.append(estimate_damage(move.type, active_pokemon.type, move.power, opponent_pokemon.type, my_attack_stage,
                                          opp_defense_stage, weather))

        move_id = int(np.argmax(damage)) 
        #  If this damage is greater than the opponents current health we knock it out
        if damage[move_id] >= opponent_pokemon.hp:
            # print("try to knock it out")
            return move_id

        # If move is super effective use it
        if damage[move_id] > 0 and TYPE_CHART_MULTIPLIER[active_pokemon.moves[move_id].type][opponent_pokemon.type] == 2.0:
            # print("Attack with supereffective")
            return move_id

        defense_type_multiplier = evaluate_matchup(active_pokemon.type, opponent_pokemon.type,
                                                   list(map(lambda m: m.type, opponent_pokemon.moves)))
        # print(defense_type_multiplier)
        if defense_type_multiplier <= 1.0:
            # Check for spike moves if spikes not setted
            if state[1].entry_hazard != PkmEntryHazard.SPIKES and opp_not_fainted_pkms > DEFAULT_PARTY_SIZE / 2:
                for i in range(DEFAULT_PKM_N_MOVES):
                    if active_pokemon.moves[i].hazard == PkmEntryHazard.SPIKES:
                        # print("Setting Spikes")
                        return i
        
        #  Use sandstorm if not setted and you have pokemons immune to that
            if weather != WeatherCondition.SANDSTORM and not self.sandstorm_used and defense_type_multiplier < 1.0:
                sandstorm_move = -1
                for i in range(DEFAULT_PKM_N_MOVES):
                    if active_pokemon.moves[i].weather == WeatherCondition.SANDSTORM:
                        sandstorm_move = i
                immune_pkms = 0
                for pkm in bench:
                    if not pkm.fainted() and pkm.type in [PkmType.GROUND, PkmType.STEEL, PkmType.ROCK]:
                        immune_pkms += 1
                if sandstorm_move != -1 and immune_pkms > 2:
                    # print("Using Sandstorm")
                    self.sandstorm_used = True
                    return sandstorm_move

            # Use hail if not setted and you have pokemons immune to that
            if weather != WeatherCondition.HAIL and not self.hail_used and defense_type_multiplier < 1.0:
                hail_move = -1
                for i in range(DEFAULT_PKM_N_MOVES):
                    if active_pokemon.moves[i].weather == WeatherCondition.HAIL:
                        hail_move = i
                immune_pkms = 0
                for pkm in bench:
                    if not pkm.fainted() and pkm.type in [PkmType.ICE]:
                        immune_pkms += 1
                if hail_move != -1 and immune_pkms > 2:
                    # print("Using Hail")
                    self.hail_used = True
                    return hail_move
            # If enemy attack and defense stage is 0 , try to use attack or defense down
            if opp_attack_stage == 0 and opp_defense_stage == 0:
                for i in range(DEFAULT_PKM_N_MOVES):
                    if active_pokemon.moves[i].target == 1 and active_pokemon.moves[i].stage != 0 and (
                            active_pokemon.moves[i].stat == PkmStat.ATTACK or active_pokemon.moves[i].stat == PkmStat.DEFENSE):
                        # print("Debuffing enemy")
                        return i

            # If spikes not set try to switch
            # print("Attacking enemy to lower his hp")
            return move_id
        
         # If we are not switch, find pokemon with resistance 
        matchup: List[float] = []
        not_fainted = False
        for pkm in bench:
            if pkm.hp == 0.0:
                matchup.append(0.0)
            else:
                not_fainted = True
                matchup.append(
                    evaluate_matchup(pkm.type, opponent_pokemon.type, list(map(lambda m: m.type, opponent_pokemon.moves))))

        best_switch = int(np.argmin(matchup))
        if not_fainted and bench[best_switch] != active_pokemon:
            # print("Switching to someone else")
            return best_switch + 4

        # If our party has no non fainted pkm, lets give maximum possible damage with current active
        # print("Nothing to do just attack")
        return move_id


