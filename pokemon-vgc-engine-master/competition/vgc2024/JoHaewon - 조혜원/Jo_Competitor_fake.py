import numpy as np
import random
from typing import List, Tuple
from copy import deepcopy

from vgc.behaviour import BattlePolicy, TeamSelectionPolicy, TeamBuildPolicy
from vgc.competition.Competitor import Competitor
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER
from vgc.datatypes.Objects import GameState, Pkm
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition, PkmEntryHazard, PkmStatus
from Jo_TeamBuildPolicies2 import Real_Good_Team
from Jo_TeamSelectionPolicies import Real_Good_Selection
from Jo_BattlePolicies8 import haewon_battlepolicies

class Jo_Competition(Competitor):


    def __init__(self, name: str = "JoHaewon_fake"):
        self._name = name
        self._battle_policy = haewon_battlepolicies()
        self._team_selection_policy = Real_Good_Selection()
        self._team_build_policy = Real_Good_Team()

    @property
    def name(self):
        return self._name
    
    @property
    def team_build_policy(self) -> TeamBuildPolicy:
        return self._team_build_policy

    @property
    def team_selection_policy(self) -> TeamSelectionPolicy:
        return self._team_selection_policy
    
    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy
