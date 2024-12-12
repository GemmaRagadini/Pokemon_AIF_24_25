from BattlePolicies import RorrisBattlePolicy
from vgc.behaviour import BattlePolicy
from vgc.competition.Competitor import Competitor


class ExampleCompetitor(Competitor):

    def __init__(self, name: str = "Rorris"):
        self._name = name
        self._battle_policy = RorrisBattlePolicy()

    @property
    def name(self):
        return self._name

    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy
