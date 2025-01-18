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
            g_copy.step([action, 99])  
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




## ESEMPIO 

class Bot4BattlePolicy(BattlePolicy):

    def __init__(self, switch_probability: float = .15, n_moves: int = DEFAULT_PKM_N_MOVES,
                 n_switches: int = DEFAULT_PARTY_SIZE):
        super().__init__()
        self.name = "Bot4BattlePolicy"

    def get_action(self, g: GameState):
        # get weather condition
        weather = g.weather.condition

        # get my pokémon
        my_team = g.teams[0]
        my_active = my_team.active
        my_attack_stage = my_team.stage[PkmStat.ATTACK]
        my_defense_stage = my_team.stage[PkmStat.DEFENSE]

        # get opp team
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_active_type = opp_active.type
        opp_attack_stage = opp_team.stage[PkmStat.ATTACK]
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]


        #priority/speed win
        attack_order = self.CanAttackFirst(my_team, opp_team, my_active, opp_active)
        if attack_order == 0:
            for move in my_active.moves:
                if move.priority and self.calculate_damage(move, my_active.type, opp_active_type, my_attack_stage, opp_defense_stage, weather) >= opp_active.hp:
                    return my_active.moves.index(move)
        
        # get most damaging move from my active pokémon
        damage: List[float] = []
        for move in my_active.moves:
            try:
                damage.append(self.calculate_damage(move, my_active.type, opp_active_type,
                                          my_attack_stage, opp_defense_stage, weather))
            except:
                #print("Something is wrong")
                damage.append(-1)

        max_move_id = int(np.argmax(damage))


        #if victory is immediately possible, defeat opponent
        if damage[max_move_id] >= opp_active.hp and attack_order >=0:
            damage_sorted = sorted(damage)
            for dmg in damage_sorted:
                if dmg >= opp_active.hp:
                    move = damage.index(dmg)
                    if my_active.moves[move].acc >= my_active.moves[max_move_id].acc and my_active.moves[move].acc >= 0.7:
                        return move

        
        #calculate survival
        survivable_turns = self.estimate_survivable_turns(my_active, opp_active, my_defense_stage, opp_attack_stage, weather) #-10 means unknown
        #calculate turns till win
        turns_to_win = self.estimate_turns_till_win(my_active, opp_active, my_attack_stage, opp_defense_stage, weather)

        #control speed if possible
        if attack_order < 1:
            for move in my_active.moves:
                if move.stat == PkmStat.SPEED and move.target == 1 and move.stage < 0 and move.prob >= 0.8:
                    if TYPE_CHART_MULTIPLIER[move.type][opp_active.type] == 0:
                        print('Stop')
                    return my_active.moves.index(move)

         #try preventing the opp from attacking
        if not(opp_active.status in [PkmStatus.SLEEP, PkmStatus.FROZEN, PkmStatus.PARALYZED, PkmStatus.CONFUSED]  or opp_team.confused):
            for move in my_active.moves:
                if move.target == 1 and move.pp > 0 and move.acc >= 0.8:
                    if (move.status == PkmStatus.FROZEN and move.prob >= 0.5) or (move.status == PkmStatus.SLEEP and move.prob >= 0.8):
                        if self.check_status_application(move.status, opp_team):
                            return my_active.moves.index(move)
                    elif (move.status == PkmStatus.PARALYZED or move.status == PkmStatus.CONFUSED) and move.prob >= 0.8:
                        if self.check_status_application(move.status, opp_team):
                            return my_active.moves.index(move)                       
                      
        #boost DEF
        def_turn_boost = []
        if my_defense_stage < 4:
            for move in my_active.moves:
                if move.stat == PkmStat.DEFENSE and move.target == 0 and move.stage > 0 and move.prob >= 0.6:
                    def_turn_boost.append(self.estimate_survivable_turns(my_active, opp_active, my_defense_stage + move.stage, opp_attack_stage, weather) - survivable_turns + (-1 if attack_order < 1 else 0))
                else:
                    def_turn_boost.append(0)

        def_boost = def_turn_boost[int(np.argmax(def_turn_boost))]      
                    
        
        
        #debuff opponent
        # turn_boost = []
        # if opp_attack_stage > MIN_STAGE:
        #     if move.stat == PkmStat.ATTACK and move.target == 1 and move.stage < 0:
        #         turn_boost.append(self.estimate_survivable_turns(my_active, opp_active, my_defense_stage, opp_attack_stage  + move.stage, weather) - survivable_turns + (-1 if attack_order < 1 else 0))
        #     else:
        #         turn_boost.append(0)

        # if turn_boost[int(np.argmax(turn_boost))] > 0:
        #     return int(np.argmax(turn_boost))

        # turn_boost = []
        # if opp_defense_stage > MIN_STAGE:
        #     if move.stat == PkmStat.DEFENSE and move.target == 1 and move.stage < 0:
        #         turn_boost.append(turns_to_win - self.estimate_turns_till_win(my_active, opp_active, my_attack_stage, opp_defense_stage + move.stage, weather) - 1)              
        #     else:
        #         turn_boost.append(0)

        # if turn_boost[int(np.argmax(turn_boost))] > 0:
        #     return int(np.argmax(turn_boost))
                    

        #can't defeat opp, try switch
        if survivable_turns > -10:
            team_chance = []
            if (turns_to_win > survivable_turns and not self.might_not_attack(opp_active)) or my_team.confused:
                for p in my_team.party:
                    team_survive = self.estimate_survivable_turns(p, opp_active, 0, opp_attack_stage, weather)
                    team_win =  self.estimate_turns_till_win(p, opp_active, 0, opp_defense_stage, weather)
                    team_chance.append(team_win)
                    team_chance.append(team_survive)

                team_pkm_1 = -1
                team_pkm_2 = -1
                if team_chance[1] > 1: 
                    team_pkm_1 = team_chance[1] -team_chance[0]
                    if team_chance[3] > 1:
                        team_pkm_2 =  team_chance[3] - team_chance[2]
                if team_pkm_1 > 0 and team_pkm_1 >= team_pkm_2 and team_pkm_1 > (turns_to_win-survivable_turns):
                    if def_boost < team_pkm_1:
                        return 4
                    elif def_boost > 0:
                        return int(np.argmax(def_turn_boost))
                if team_pkm_2 > 0 and team_pkm_1 < team_pkm_2 and team_pkm_2 > (turns_to_win-survivable_turns):
                    if def_boost < team_pkm_2:
                        return 5
                    else:
                        return int(np.argmax(def_turn_boost))

        #boost atk if it pays off
        if def_boost > 0:
            if def_boost == 1 and my_defense_stage > 2 and damage[max_move_id]/opp_active.hp >= 0.20:
                return max_move_id
            return int(np.argmax(def_turn_boost))
        atk_turn_boost = []
        if my_attack_stage < 4 and survivable_turns > -10:
            for move in my_active.moves:
                if move.stat == PkmStat.ATTACK and move.target == 0 and move.stage > 0 and move.prob >= 0.6:
                    atk_turn_boost.append(turns_to_win - self.estimate_turns_till_win(my_active, opp_active, my_attack_stage + move.stage, opp_defense_stage, weather) - 1)
                else:
                    atk_turn_boost.append(0)

            if atk_turn_boost[int(np.argmax(atk_turn_boost))] > 0:
                return int(np.argmax(atk_turn_boost))

        return max_move_id

    def estimate_survivable_turns(self, pkm:Pkm, opp:Pkm, own_def_stage:int, opp_atk_stage:int, weather):
        turns:int = 0
        hp:float = pkm.hp
        
        [move_ids, opp_dmg] = self.get_max_damage_moves_sorted(opp, pkm, opp_atk_stage, own_def_stage, weather)
        max_dmg_move = move_ids[0] 
        max_dmg_move_index = 0
        pp_cost = [0, 0, 0, 0]
        if opp.moves[max_dmg_move].name == None:
            return -10
        while hp > 0 and turns < 20:
            turns += 1
            if opp.moves[max_dmg_move].pp - pp_cost[max_dmg_move] > 0:
                hp -= opp_dmg[max_dmg_move]
                pp_cost[max_dmg_move] += 1
            else:
                max_dmg_move_index += 1
                max_dmg_move = move_ids[max_dmg_move]
                hp -= opp_dmg[max_dmg_move]
                pp_cost[max_dmg_move] += 1
            if (opp.moves[max_dmg_move].stat == PkmStat.DEFENSE and opp.moves[max_dmg_move].target == 1) or (opp.moves[max_dmg_move].stat == PkmStat.ATTACK and opp.moves[max_dmg_move].target == 0):
                [move_ids, opp_dmg] = self.get_max_damage_moves_sorted(opp, pkm, opp_atk_stage, own_def_stage, weather)

        return turns if turns < 20 else -10
    
    def estimate_turns_till_win(self, pkm:Pkm, opp:Pkm, own_atk_stage:int, opp_def_stage:int, weather) -> int:
        return self.estimate_survivable_turns(opp, pkm, opp_def_stage, own_atk_stage, weather)
            
    def might_not_attack(self, pkm:Pkm):
        if pkm.status in [PkmStatus.CONFUSED, PkmStatus.FROZEN, PkmStatus.PARALYZED, PkmStatus.SLEEP]:
            return True
        else:
            return False
    
    def calculate_damage(self, move: PkmMove, pkm_type: PkmType, opp_pkm_type: PkmType, attack_stage: int, defense_stage: int, weather: WeatherCondition) -> float:
        if move.pp <= 0:
            move = Struggle
        
        fixed_damage = move.fixed_damage
        if fixed_damage > 0. and TYPE_CHART_MULTIPLIER[move.type][opp_pkm_type] > 0.:
            damage = fixed_damage
        else:
            stab = 1.5 if move.type == pkm_type else 1.
            if (move.type == PkmType.WATER and weather == WeatherCondition.RAIN) or (move.type == PkmType.FIRE and weather == WeatherCondition.SUNNY):
                weather = 1.5
            elif (move.type == PkmType.WATER and weather == WeatherCondition.SUNNY) or (move.type == PkmType.FIRE and weather == WeatherCondition.RAIN):
                weather = .5
            else:
                weather = 1.       
        
            stage_level = attack_stage - defense_stage
            stage = (stage_level + 2.) / 2 if stage_level >= 0. else 2. / (np.abs(stage_level) + 2.)
            multiplier = TYPE_CHART_MULTIPLIER[move.type][opp_pkm_type] if move != Struggle else 1.0
            damage = multiplier * stab * weather * stage * move.power
        return round(damage)
    
    #-1 lower speed, 0 same speed, or enemy has prio, 1 higher speed and opponent has no prio
    def CanAttackFirst(self, my_team:PkmTeam, opp_team:PkmTeam, my_active:Pkm, opp_active:Pkm) -> int:
        """
        Get attack order for this turn.

        :return: -2 opp faster and has priority, -1 opp faster, 1 self faster and no opp prio, 0 same speed, 0.5 if faster but opp prio
        """
        speed0 = my_team.stage[PkmStat.SPEED]
        speed1 = opp_team.stage[PkmStat.SPEED]

        opp_might_act_earlier = False
        for opp_poss_act in opp_active.moves:
            if opp_poss_act.priority:
                opp_might_act_earlier = True

        if speed1 > speed0:
            if opp_might_act_earlier:
                return -2
            return -1
        if speed0 > speed1 and not opp_might_act_earlier:
            return 1
        if speed0 > speed1 and opp_might_act_earlier:
            return 0.5
        else:
            return 0

    def get_switch_opp_greedy(self, my_team:PkmTeam, opp_active:Pkm, opp_move_id: int, opp_attack_stage:int, weather):       
        #evaluate wich of my pokemon would get hurt less
        best_state = -10
        index = 0
        for pkm in my_team.party:
            if pkm != my_team.active and pkm.hp > 0:
                state = (pkm.hp - self.calculate_damage(opp_active.moves[opp_move_id],  opp_active.type, pkm.type, opp_attack_stage, 0, weather)) / pkm.max_hp
                if state > best_state:
                    index = my_team.party.index(pkm)
                    best_state = state
        
        if best_state > 0 and index > 0:
            return index
        else: 
            return 0

    def get_possible_damages(self, attacker: Pkm, defender: Pkm, attack_stage: int, defense_stage: int, weather) -> list[float]:
        damage: List[float] = []
        for move in attacker.moves:
            try:
                damage.append(self.calculate_damage(move, attacker.type, defender.type, attack_stage, defense_stage, weather))
                
            except:               
                damage.append(-1)
                pass
        return damage

    def get_max_damage_move(self, attacker: Pkm, defender: Pkm, attack_stage, defense_stage, weather) -> list[int, float]:
        damage = self.get_possible_damages(attacker, defender, attack_stage, defense_stage, weather)

        move_id = int(np.argmax(damage))

        return [move_id, damage[move_id]]
    
    def get_max_damage(self, attacker: Pkm, defender: Pkm, attack_stage, defense_stage, weather) -> int:
        damage = self.get_possible_damages(attacker, defender, attack_stage, defense_stage, weather)

        move_id = int(np.argmax(damage))

        return damage[move_id]

    def get_max_damage_moves_sorted(self, attacker: Pkm, defender: Pkm, attack_stage, defense_stage, weather) -> list[list[int], list[float]]:
        damage = self.get_possible_damages(attacker, defender, attack_stage, defense_stage, weather)

        damage_set = [[i, damage[i]] for i in range(0, 4)]
        damage_sorted = sorted(damage_set, key=itemgetter(1), reverse=True)
        move_ids = [damage_sorted[0][0], damage_sorted[1][0], damage_sorted[2][0], damage_sorted[3][0]]

        return [move_ids, damage]

    def check_status_application(self, status: PkmStatus, opp_team: PkmTeam) -> bool:
        pkm = opp_team.active

        if status == PkmStatus.PARALYZED and pkm.type != PkmType.ELECTRIC and pkm.type != PkmType.GROUND and pkm.status != PkmStatus.PARALYZED:
            #print("Opponent can be paralyzed!")
            #pkm.status = PkmStatus.PARALYZED
            return True
        elif status == PkmStatus.POISONED and pkm.type != PkmType.POISON and pkm.type != PkmType.STEEL and pkm.status != PkmStatus.POISONED:
            #print("Opponent can be poisoned!")
            return True
        elif status == PkmStatus.BURNED and pkm.type != PkmType.FIRE and pkm.status != PkmStatus.BURNED:
            #print("Opponent can be burned!")
            return True
        elif status == PkmStatus.SLEEP and pkm.status != PkmStatus.SLEEP:
            #print("Opponent can be put asleep!")
            return True
        elif status == PkmStatus.FROZEN and pkm.type != PkmType.ICE and pkm.status != PkmStatus.FROZEN:
            #print("Opponent can be frozen solid!")
            return True
        elif not opp_team.confused:
            #print("Opponent can be confused!")
            return True
        
        return False
    
    def CheckSwitchWorstCase(self, my_team:PkmTeam, my_pkm: Pkm, opp_pkm: Pkm, opp_attack_stage: int, weather: WeatherCondition) -> float:
        damage = 0
        for move in opp_pkm.moves:
            move_damage = self.calculate_damage(move, opp_pkm.type, my_pkm.type, opp_attack_stage, 0 , weather)
            if move_damage > damage:
                damage = move_damage

        #look out for entry hazards and weather damage
        spikes = my_team.entry_hazard[PkmEntryHazard.SPIKES]
        if spikes > 0:
            damage += STATE_DAMAGE if spikes <= 1 else SPIKES_2 if spikes == 2 else SPIKES_3
        if weather == WeatherCondition.SANDSTORM and (my_pkm.type != PkmType.ROCK and my_pkm.type != PkmType.GROUND and my_pkm.type != PkmType.STEEL):
            damage += STATE_DAMAGE
        elif self.weather.condition == WeatherCondition.HAIL and (my_pkm.type != PkmType.ICE):
            damage += STATE_DAMAGE

        return damage
    




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


# My Battle Policy
class DBaziukBattlePolicy(BattlePolicy):

    def init(self, switch_probability: float = .15, n_moves: int = DEFAULT_PKM_N_MOVES,
                 n_switches: int = DEFAULT_PARTY_SIZE):
        super().init()
        self.hail_used = False
        self.sandstorm_used = False
        self.name = "DBaziukBattlePolicy"

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g: GameState) -> int:
        # print("It's me")
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
        damage: List[float] = []
        for move in my_active.moves:
            damage.append(estimate_damage(move.type, my_active.type, move.power, opp_active.type, my_attack_stage,
                                          opp_defense_stage, weather))

        # get most damaging move
        move_id = int(np.argmax(damage))

        #  If this damage is greater than the opponents current health we knock it out
        if damage[move_id] >= opp_active.hp:
            # print("try to knock it out")
            return move_id

        # If move is super effective use it
        if damage[move_id] > 0 and TYPE_CHART_MULTIPLIER[my_active.moves[move_id].type][opp_active.type] == 2.0:
            # print("Attack with supereffective")
            return move_id

        defense_type_multiplier = evaluate_matchup(my_active.type, opp_active.type,
                                                   list(map(lambda m: m.type, opp_active.moves)))
        # print(defense_type_multiplier)
        if defense_type_multiplier <= 1.0:
            # Check for spike moves if spikes not setted
            if opp_team.entry_hazard != PkmEntryHazard.SPIKES and opp_not_fainted_pkms > DEFAULT_PARTY_SIZE / 2:
                for i in range(DEFAULT_PKM_N_MOVES):
                    if my_active.moves[i].hazard == PkmEntryHazard.SPIKES:
                        # print("Setting Spikes")
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
                    # print("Using Sandstorm")
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
                    # print("Using Hail")
                    self.hail_used = True
                    return hail_move

            # If enemy attack and defense stage is 0 , try to use attack or defense down
            if opp_attack_stage == 0 and opp_defense_stage == 0:
                for i in range(DEFAULT_PKM_N_MOVES):
                    if my_active.moves[i].target == 1 and my_active.moves[i].stage != 0 and (
                            my_active.moves[i].stat == PkmStat.ATTACK or my_active.moves[i].stat == PkmStat.DEFENSE):
                        # print("Debuffing enemy")
                        return i

            # If spikes not set try to switch
            # print("Attacking enemy to lower his hp")
            return move_id

        # If we are not switch, find pokemon with resistance 
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
            # print("Switching to someone else")
            return best_switch + 4

        # If our party has no non fainted pkm, lets give maximum possible damage with current active
        # print("Nothing to do just attack")
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



class AggressiveMinimaxWithKiller(BattlePolicy):
    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        self.name = "Aggressive Minimax with Killer Moves"
        self.killer_moves = {depth: [] for depth in range(max_depth + 1)}

    def aggressive_eval(self, state):
        """
        Funzione di valutazione euristica basata sulla politica aggressiva.
        """
        mine = state.teams[0].active
        opp = state.teams[1].active

        # Penalità per lo stato corrente (switch obbligatorio)
        if opp.is_super_effective_against(mine):
            return -float('inf')  # Forza lo switch

        # Miglior danno possibile
        best_damage = 0
        for move in mine.moves:
            if move.pp > 0 and not move.has_negative_effect_on_user():
                damage = calculate_damage(move, mine, opp)
                if mine.speed > opp.speed:  # Bonus velocità
                    damage *= 1.1
                best_damage = max(best_damage, damage)

        return best_damage

    def minimax(self, state, depth, alpha, beta, is_maximizing_player):
        """
        Minimax con pruning alpha-beta e killer move heuristic.
        """
        if depth == 0 or state.is_terminal():
            return self.aggressive_eval(state), None

        killer_key = (state.hash(), depth)
        killer_action = self.killer_moves.get(depth, [])

        if is_maximizing_player:
            max_eval = float('-inf')
            best_action = None

            moves = list(range(DEFAULT_N_ACTIONS))
            # Ordina mosse per killer moves e priorità euristica
            moves = sorted(moves, key=lambda m: m in killer_action, reverse=True)

            for action in moves:
                g_copy = deepcopy(state)
                g_copy.step([action, 99])  # Azione avversaria casuale
                eval_score, _ = self.minimax(g_copy, depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    if action not in killer_action:
                        killer_action.append(action)
                        if len(killer_action) > 2:  # Limita a 2 killer moves
                            killer_action.pop(0)
                    break

            self.killer_moves[depth] = killer_action
            return max_eval, best_action

        else:
            min_eval = float('inf')
            best_action = None

            moves = list(range(DEFAULT_N_ACTIONS))
            # Ordina mosse per killer moves e priorità euristica
            moves = sorted(moves, key=lambda m: m in killer_action, reverse=True)

            for action in moves:
                g_copy = deepcopy(state)
                g_copy.step([99, action])  # Azione casuale del giocatore
                eval_score, _ = self.minimax(g_copy, depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                beta = min(beta, eval_score)
                if beta <= alpha:
                    if action not in killer_action:
                        killer_action.append(action)
                        if len(killer_action) > 2:  # Limita a 2 killer moves
                            killer_action.pop(0)
                    break

            self.killer_moves[depth] = killer_action
            return min_eval, best_action

    def get_action(self, state):
        """
        Trova la migliore azione utilizzando Minimax con killer move heuristic.
        """
        _, best_action = self.minimax(state, self.max_depth, float('-inf'), float('inf'), True)
        return best_action if best_action is not None else 0



class AggressiveMinimaxWithSwitchAndKiller(BattlePolicy):
    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        self.name = "Aggressive Minimax with Switch and Killer Moves"
        self.killer_moves = {depth: [] for depth in range(max_depth + 1)}

    def aggressive_eval(self, state):
        """
        Funzione di valutazione euristica basata sulla politica aggressiva.
        """
        mine = state.teams[0].active
        opp = state.teams[1].active

        # Penalità per lo stato corrente (switch obbligatorio)
        if opp.is_super_effective_against(mine):
            return -float('inf')  # Forza lo switch

        # Miglior danno possibile
        best_damage = 0
        for move in mine.moves:
            if move.pp > 0 and not move.has_negative_effect_on_user():
                damage = calculate_damage(move, mine, opp)
                if mine.speed > opp.speed:  # Bonus velocità
                    damage *= 1.1
                best_damage = max(best_damage, damage)

        return best_damage

    def get_switch_action(self, state):
        """
        Restituisce lo switch a un Pokémon che sia super efficace contro l'avversario.
        """
        opp = state.teams[1].active
        for i, pokemon in enumerate(state.teams[0].reserve):
            if pokemon.is_super_effective_against(opp):
                return i + 1  # L'azione di switch è rappresentata dagli indici successivi alle mosse
        return None

    def minimax(self, state, depth, alpha, beta, is_maximizing_player):
        """
        Minimax con pruning alpha-beta, killer move heuristic e switch.
        """
        if depth == 0 or state.is_terminal():
            return self.aggressive_eval(state), None

        killer_key = (state.hash(), depth)
        killer_action = self.killer_moves.get(depth, [])

        if is_maximizing_player:
            max_eval = float('-inf')
            best_action = None

            # Genera le azioni: mosse + eventuale switch
            moves = list(range(DEFAULT_N_ACTIONS))
            switch_action = self.get_switch_action(state)
            if switch_action is not None:
                moves.append(switch_action)

            # Ordina mosse per killer moves e priorità euristica
            moves = sorted(moves, key=lambda m: m in killer_action, reverse=True)

            for action in moves:
                g_copy = deepcopy(state)
                if action >= DEFAULT_N_ACTIONS:  # Azione di switch
                    g_copy.switch([action - DEFAULT_N_ACTIONS, 99])
                else:  # Azione normale
                    g_copy.step([action, 99])

                eval_score, _ = self.minimax(g_copy, depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    if action not in killer_action:
                        killer_action.append(action)
                        if len(killer_action) > 2:  # Limita a 2 killer moves
                            killer_action.pop(0)
                    break

            self.killer_moves[depth] = killer_action
            return max_eval, best_action

        else:
            min_eval = float('inf')
            best_action = None

            # Genera le azioni: mosse + eventuale switch
            moves = list(range(DEFAULT_N_ACTIONS))
            switch_action = self.get_switch_action(state)
            if switch_action is not None:
                moves.append(switch_action)

            # Ordina mosse per killer moves e priorità euristica
            moves = sorted(moves, key=lambda m: m in killer_action, reverse=True)

            for action in moves:
                g_copy = deepcopy(state)
                if action >= DEFAULT_N_ACTIONS:  # Azione di switch
                    g_copy.switch([99, action - DEFAULT_N_ACTIONS])
                else:  # Azione normale
                    g_copy.step([99, action])

                eval_score, _ = self.minimax(g_copy, depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                beta = min(beta, eval_score)
                if beta <= alpha:
                    if action not in killer_action:
                        killer_action.append(action)
                        if len(killer_action) > 2:  # Limita a 2 killer moves
                            killer_action.pop(0)
                    break

            self.killer_moves[depth] = killer_action
            return min_eval, best_action

    def get_action(self, state):
        """
        Trova la migliore azione utilizzando Minimax con switch e killer move heuristic.
        """
        _, best_action = self.minimax(state, self.max_depth, float('-inf'), float('inf'), True)
        return best_action if best_action is not None else 0
