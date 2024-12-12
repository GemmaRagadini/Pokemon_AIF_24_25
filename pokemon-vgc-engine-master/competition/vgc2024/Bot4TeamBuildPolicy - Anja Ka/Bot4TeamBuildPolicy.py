from copy import deepcopy
from typing import List

import numpy as np
from operator import itemgetter
from scipy.optimize import linprog
from torch import nn

from vgc.balance.meta import MetaData
from vgc.behaviour import TeamBuildPolicy, BattlePolicy
from vgc.behaviour.BattlePolicies import TypeSelector
from vgc.competition.StandardPkmMoves import STANDARD_MOVE_ROSTER
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, MAX_HIT_POINTS
from vgc.datatypes.Objects import Pkm, PkmTemplate, PkmFullTeam, PkmRoster, PkmTeam, PkmMove
from vgc.datatypes.Types import N_TYPES, N_STATUS, N_ENTRY_HAZARD
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.util.Encoding import one_hot
from vgc.datatypes.Types import PkmStatus, PkmStat


class Bot4TeamBuilder(TeamBuildPolicy):


    def __init__(self):
        self.roster = None

    """
    At the beginning of a championship, or during a meta-game balance competition, 
    set_roster is called providing the information about the available roster
    """
    def set_roster(self, roster: PkmRoster, ver: int = 0):
        self.roster = roster #List[PkmTemplate]


    def get_action(self, meta: MetaData) -> PkmFullTeam:
        hp_threshold = 150
        #select only pokemon that don't get one-shotted
        pre_pre_selection: List[PkmTemplate] = []
        for pt in self.roster:
            move_types = []
            if pt.max_hp < hp_threshold:
                continue
            for move in pt.moves:
                if move.power > 0 or move.fixed_damage > 0:
                    move_types.append(move.type)
            if len(move_types) < 2:
                continue
            pre_pre_selection.append(pt)

        #make sure enough pokemon for a full team are left
        while len(pre_pre_selection) < 3:
            hp_threshold -= 30
            for pt in self.roster:
                if not pre_pre_selection.__contains__(pt) and pt.max_hp >= hp_threshold:
                    pre_pre_selection.append(pt)
                    break
        
        stat_pre_selection: list[PkmTemplate] = []
        for pt in pre_pre_selection:           
            if not (self.has_own_atk_up(pt) or self.has_opp_speed_down(pt)):
                continue
            stat_pre_selection.append(pt)

        if len(stat_pre_selection) < 3:
            for pt in pre_pre_selection:
                 if len(stat_pre_selection) < 3 and not stat_pre_selection.__contains__(pt):
                    stat_pre_selection.append(pt)

        #sort by overall damage
        pre_selection: List[List[float, PkmTemplate]] = []
        for pt in stat_pre_selection:
            pre_selection.append([self.power_over_all_moves(pt), pt, pt.type])
        pre_selection = sorted(pre_selection, key=itemgetter(0), reverse=True)    

        pre_pre_selection_sort = []
        for pt in pre_pre_selection:
            pre_pre_selection_sort.append([self.power_over_all_moves(pt), pt])
        pre_pre_selection_sort = sorted(pre_pre_selection_sort, key=itemgetter(0), reverse=True)     

        #select mightiest of each type that controls speed
        pkm_types = []
        selection: List[PkmTemplate] = []
        for pkm in pre_selection:
            if (not(pkm[2] in pkm_types)) and self.has_opp_speed_down(pkm[1]) and pkm[0] >= 150:
                pkm_types.append(pkm[1].type)
                selection.append(pkm[1])

        #select mightiest of each type
        if len(selection) < 3:
            selection: List[PkmTemplate] = []
            for pkm in pre_pre_selection_sort:
                #print(pkm[1], 'power:', pkm[0])
                if not pkm[1].type in pkm_types:
                    pkm_types.append(pkm[1].type)
                    selection.append(pkm[1])

        #select from candidates
        team: List[Pkm] = []
        for pt in range(0, 3):
            team.append(selection[pt].gen_pkm([0, 1, 2, 3]))
        return PkmFullTeam(team)

    

    def power_over_all_moves(self, pkm:Pkm):
        power = 0
        for move in pkm.moves:
            if move.fixed_damage > 0:
                power += move.fixed_damage * move.acc
            else:
                stab = 1.5 if move.type == pkm.type else 1.
                power += move.power * stab * move.acc
        return power
    
    def max_move_damage(self, pkm:Pkm):
        power = 0
        for move in pkm.moves:
            if move.fixed_damage > 0 and power < move.fixed_damage * move.acc:
                power = move.fixed_damage * move.acc
            else:
                stab = 1.5 if move.type == pkm.type else 1.
                if move.power * stab * move.acc > power:
                    power = move.power * stab * move.acc
        return power
    
    def has_opp_stage_down(self, stat:PkmStat, pkm:PkmTemplate):
        for move in pkm.moves:
            if move.stat == stat and move.stage < 0 and move.target == 1 and move.prob >= 0.6:
                return True
            
        return False
    
    def has_own_stage_major_up(self, stat:PkmStat, pkm:PkmTemplate):
        for move in pkm.moves:
            if move.stat == stat and move.stage > 1 and move.prob >= 0.6:
                return True
            
        return False

    def has_opp_speed_down(self, pkm:PkmTemplate):
        return self.has_opp_stage_down(PkmStat.SPEED, pkm)
    
    def has_opp_def_down(self, pkm:PkmTemplate):
        return self.has_opp_stage_down(PkmStat.DEFENSE, pkm)
    
    def has_opp_atk_down(self, pkm:PkmTemplate):
        return self.has_opp_stage_down(PkmStat.ATTACK, pkm)
    
    def has_own_def_up(self, pkm:PkmTemplate):
        return self.has_own_stage_major_up(PkmStat.DEFENSE, pkm)
    
    def has_own_atk_up(self, pkm:PkmTemplate):
        return self.has_own_stage_major_up(PkmStat.ATTACK, pkm)