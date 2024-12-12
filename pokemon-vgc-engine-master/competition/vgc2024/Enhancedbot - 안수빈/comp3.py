from battle3 import EnhancedBattlePolicy
from vgc.behaviour import BattlePolicy
from vgc.competition.Competitor import Competitor

class MyCompetitor(Competitor):

    def __init__(self, name: str = "EnhancedBot"):
        self._name = name
        self._battle_policy = EnhancedBattlePolicy()

    @property
    def name(self):
        return self._name

    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy
