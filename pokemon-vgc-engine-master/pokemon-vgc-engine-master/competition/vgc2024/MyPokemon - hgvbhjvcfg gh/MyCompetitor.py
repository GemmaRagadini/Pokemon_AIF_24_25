from vgc.behaviour import BattlePolicy, TeamBuildPolicy
from vgc.competition.Competitor import Competitor

from MyBattlePolicy import MyBattlePolicy
from MyTeamBuildPolicy import MyTeamBuilder

class MyCompetitor(Competitor):

    def __init__(self, name: str = "MyComprtitor"):
        self._name = name
        self._battle_policy = MyBattlePolicy()
        self._team_build_policy = MyTeamBuilder()

    @property
    def name(self):
        return self._name

    @property
    def team_build_policy(self) -> TeamBuildPolicy:
        return self._team_build_policy
    
    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy