from MyBattlePoliciesMR import KidMeBot
from vgc.behaviour import BattlePolicy

from vgc.competition.Competitor import Competitor


class KidMeCompetitor(Competitor):

    def __init__(self, name: str = "KidMe"):
        self._name = name
        self._battle_policy = KidMeBot()

    @property
    def name(self):
        return self._name

    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy
