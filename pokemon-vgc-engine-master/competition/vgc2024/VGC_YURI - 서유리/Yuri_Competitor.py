import argparse
from Yuri_BattlePolicies import *
from Yuri_TeamBuildPolicies import *
from vgc.competition.Competitor import Competitor

from vgc.behaviour import BattlePolicy, TeamSelectionPolicy, TeamBuildPolicy
from vgc.behaviour.BattlePolicies import RandomPlayer, TerminalPlayer
from vgc.behaviour.TeamBuildPolicies import IndividualPkmCounter, MaxPkmCoverage, RandomTeamBuilder, FixedTeamBuilder



class BattleCompetitor(Competitor):

    def __init__(self, name: str = "Yuri_Battle"):
        self._name = name
        self._battle_policy = BattleTrackBot()

    @property
    def name(self):
        return self._name

    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy
    
    

    

class ChampionCompetitor(Competitor):

    def __init__(self, name: str = "Yuri_Championship"):
        self._name = name
        self._battle_policy = ChampionshipTrackBot()
        self._team_build_policy = MaxPkmCoverage_UCB()

    @property
    def name(self):
        return self._name

    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy
    
    @property
    def team_build_policy(self) -> TeamBuildPolicy:
        return self._team_build_policy


