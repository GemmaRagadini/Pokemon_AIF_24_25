from vgc.behaviour.TeamBuildPolicies import RandomTeamBuilder
from vgc.behaviour import BattlePolicy, TeamBuildPolicy
from vgc.behaviour.BattlePolicies import RandomPlayer, TerminalPlayer
from vgc.competition.Competitor import Competitor

from LucyPolicies import LucyPolicy   ## 배틀 전략 
from LucyTeamBuild import LucyBuildPolicy  ## 팀빌딩 전략 

class ExampleCompetitor(Competitor):
    def __init__(self, name: str = "Lucybot"):
        self._name = name
        self._battle_policy = LucyPolicy()   ## 배틀 전략 
        self._team_build_policy = LucyBuildPolicy()  ## 팀빌딩 전략 

## identify agent 
    @property
    def name(self):
        return self._name  ## 이름 

    # strategy 
    @property
    def team_build_policy(self) -> TeamBuildPolicy:
        return self._team_build_policy

    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy

