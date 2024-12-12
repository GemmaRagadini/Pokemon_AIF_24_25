from vgc.behaviour import BattlePolicy, TeamSelectionPolicy, TeamBuildPolicy
from vgc.behaviour.BattlePolicies import RandomPlayer, TerminalPlayer ,OneTurnLookahead, Minimax
from vgc.behaviour.TeamBuildPolicies import TerminalTeamBuilder, RandomTeamBuilder
from vgc.behaviour.TeamSelectionPolicies import FirstEditionTeamSelectionPolicy
from vgc.competition.Competitor import Competitor
from my_BattlePolicy import TeraBattlePolicy
#from NiBot_BattlePolicy import NiBot
from vgc.datatypes.Objects import Pkm, PkmTemplate, PkmFullTeam, PkmRoster, PkmTeam, PkmMove
from vgc.balance.meta import MetaData
import numpy as np
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, MAX_HIT_POINTS
from typing import List, Optional, Tuple, Union
#from my_TeamBuilder import MyTeamBuilder


class TeraCompetitor(Competitor):
    def __init__(self, name: str = "Example"):
        self._name = name
        self._battle_policy = TeraBattlePolicy()
        self._team_selection_policy = FirstEditionTeamSelectionPolicy()
        self._team_build_policy = RandomTeamBuilder()

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


class TerminalExampleCompetitor(TeraCompetitor):

    def __init__(self, name: str = "TerminalPlayer"):
        super().__init__(name)
        self._battle_policy = TerminalPlayer()
        self._team_selection_policy = TeamSelectionPolicy()
        self._team_build_policy = TerminalTeamBuilder()

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


"""
class GUIExampleCompetitor(ExampleCompetitor):

    def __init__(self, name: str = "GUIPlayer"):
        self._battle_policy = GUIPlayer()
        self._team_selection_policy = GUITeamSelection()
        self._team_build_policy = GUITeamBuilder()

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
"""

class RandomTeamBuilder(TeamBuildPolicy):
    """
    Agent that selects teams randomly.
    """

    def __init__(self):
        self.roster = None

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        self.roster = roster

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        n_pkms = len(self.roster)
        members = np.random.choice(n_pkms, 3, False)
        pre_selection: List[PkmTemplate] = [self.roster[i] for i in members]
        team: List[Pkm] = []
        for pt in pre_selection:
            moves: List[int] = np.random.choice(DEFAULT_PKM_N_MOVES, DEFAULT_PKM_N_MOVES, False)
            team.append(pt.gen_pkm(moves))
        return PkmFullTeam(team)