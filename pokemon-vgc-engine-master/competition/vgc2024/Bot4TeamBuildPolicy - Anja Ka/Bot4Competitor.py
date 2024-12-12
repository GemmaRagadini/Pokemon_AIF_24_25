from vgc.behaviour import BattlePolicy, TeamSelectionPolicy, TeamBuildPolicy
from vgc.behaviour.TeamSelectionPolicies import FirstEditionTeamSelectionPolicy
from vgc.competition.Competitor import Competitor
from Bot4TeamBuildPolicy import Bot4TeamBuilder
from Bot4BattlePolicy import Bot4BattlePolicy

class Bot4Competitor(Competitor):

    def __init__(self, name: str = "Bot 4"):
        self._name = name
        self._battle_policy = Bot4BattlePolicy()
        #self._team_selection_policy = FirstEditionTeamSelectionPolicy()
        self._team_build_policy = Bot4TeamBuilder()

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

