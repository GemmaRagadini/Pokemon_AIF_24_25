from vgc.behaviour import BattlePolicy, TeamSelectionPolicy, TeamBuildPolicy
from vgc.behaviour.BattlePolicies import RandomPlayer, TerminalPlayer, MyMinimax, MyMinimaxWithAlphaBeta, MyMonteCarlo, MyMonteCarloWithMinimax
from vgc.behaviour.TeamBuildPolicies import TerminalTeamBuilder, RandomTeamBuilder
from vgc.behaviour.TeamSelectionPolicies import FirstEditionTeamSelectionPolicy
from vgc.competition.Competitor import Competitor

# aggiunto
class MyCompetitor(Competitor): 

    def __init__(self, name: str = "My Example"):
        self._name = name
        self._battle_policy =  MyMinimaxWithAlphaBeta()
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


class ExampleCompetitor(Competitor):

    def __init__(self, name: str = "Example"):
        self._name = name
        self._battle_policy = RandomPlayer()
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


class TerminalExampleCompetitor(ExampleCompetitor):

    def __init__(self, name: str = "TerminalPlayer"):
        super().__init__(name)
        self._battle_policy = TerminalPlayer()
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


