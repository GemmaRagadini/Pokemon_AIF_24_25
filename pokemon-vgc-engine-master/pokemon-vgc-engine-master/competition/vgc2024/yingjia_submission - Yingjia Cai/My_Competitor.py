from Battle_agent_eval import deep_q_agent
from vgc.behaviour import BattlePolicy
from vgc.competition.Competitor import Competitor


class ExampleCompetitor(Competitor):

    def __init__(self, name: str = "DQN_agent"):
        self._name = name
        self._battle_policy = deep_q_agent(model_dict="model_random_team_self_play.pth")

    @property
    def name(self):
        return self._name

    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy
