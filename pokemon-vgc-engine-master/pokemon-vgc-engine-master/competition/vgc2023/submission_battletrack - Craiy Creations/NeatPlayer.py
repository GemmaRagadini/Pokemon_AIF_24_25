import os
import pickle

from vgc.behaviour.BattlePolicies import BattlePolicy
from vgc.datatypes.Constants import *
from vgc.datatypes.Types import *

# Constants
ALL_POKEMON_TYPES = len(PkmType)
ALL_POKEMON_STATS = len(PkmStat)
ALL_POKEMON_STATUS = len(PkmStatus)
ALL_WEATHER = len(WeatherCondition)


class NeatPlayer(BattlePolicy):

    def __init__(self, net=None):
        if net == None:
            net = self.load_from_file(os.path.join(os.path.dirname(__file__), "network.pkl"))
        self.net = net

    def load_from_file(self, path):
        with open(path, "rb") as file:
            net = pickle.load(file)
        return net

    def get_action(self, g) -> int:
        input_nodes = self._generate_input_vector(g)
        predictions = self.net.activate(input_nodes)
        return predictions.index(max(predictions))

    def _generate_input_vector(self, gamestate):
        own_team = gamestate.teams[0]
        enemy_team = gamestate.teams[1]

        return [
            int(enemy_team.active.type) / ALL_POKEMON_TYPES,
            enemy_team.active.hp / MAX_HIT_POINTS,
            int(enemy_team.stage[PkmStat.DEFENSE]) / ALL_POKEMON_STATS,
            int(enemy_team.active.status) / ALL_POKEMON_STATUS,
            own_team.active.type / ALL_POKEMON_TYPES,
            own_team.active.hp / MAX_HIT_POINTS,
            int(own_team.stage[PkmStat.ATTACK]) / ALL_POKEMON_STATS,
            int(own_team.active.status) / ALL_POKEMON_STATUS,
            int(own_team.party[0].type) / ALL_POKEMON_TYPES,
            own_team.party[0].hp / MAX_HIT_POINTS,
            int(own_team.party[1].type) / ALL_POKEMON_TYPES,
            own_team.party[1].hp / MAX_HIT_POINTS,
            int(own_team.active.moves[0].type) / ALL_POKEMON_TYPES,
            own_team.active.moves[0].power / MOVE_POWER_MAX,
            int(own_team.active.moves[0].status) / ALL_POKEMON_STATUS,
            own_team.active.moves[0].acc,
            int(own_team.active.moves[1].type) / ALL_POKEMON_TYPES,
            own_team.active.moves[1].power / MOVE_POWER_MAX,
            int(own_team.active.moves[1].status) / ALL_POKEMON_STATUS,
            own_team.active.moves[1].acc,
            int(own_team.active.moves[2].type) / ALL_POKEMON_TYPES,
            own_team.active.moves[2].power / MOVE_POWER_MAX,
            int(own_team.active.moves[2].status) / ALL_POKEMON_STATUS,
            own_team.active.moves[2].acc,
            int(own_team.active.moves[3].type) / ALL_POKEMON_TYPES,
            own_team.active.moves[3].power / MOVE_POWER_MAX,
            int(own_team.active.moves[3].status) / ALL_POKEMON_STATUS,
            own_team.active.moves[3].acc,
            int(gamestate.weather.condition) / ALL_WEATHER
        ]

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass
