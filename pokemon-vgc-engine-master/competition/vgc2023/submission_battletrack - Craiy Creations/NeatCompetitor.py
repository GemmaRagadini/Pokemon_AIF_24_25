from NeatPlayer import NeatPlayer
from vgc.behaviour import BattlePolicy
from vgc.competition.Competitor import Competitor


class NeatCompetitor(Competitor):

    def __init__(self, name: str = "Example"):
        self._name = name
        self._battle_policy = NeatPlayer()

    @property
    def name(self):
        return self._name

    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy
