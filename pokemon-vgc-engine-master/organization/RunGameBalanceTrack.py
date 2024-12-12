import argparse
from multiprocessing.connection import Client

from vgc.balance.meta import StandardMetaData, BaseMetaEvaluator
from vgc.balance.restriction import VGCDesignConstraints
from vgc.behaviour import BattlePolicy
from vgc.behaviour.BattlePolicies import TypeSelector
from vgc.competition.Competitor import CompetitorManager, Competitor
from vgc.competition.StandardPkmMoves import STANDARD_MOVE_ROSTER
from vgc.ecosystem.GameBalanceEcosystem import GameBalanceEcosystem
from vgc.network.ProxyCompetitor import ProxyCompetitor
from vgc.util.generator.PkmRosterGenerators import RandomPkmRosterGenerator


class SurrogateCompetitor(Competitor):

    def __init__(self, name: str = "Example"):
        self._name = name
        self._battle_policy = TypeSelector()

    @property
    def name(self):
        return self._name

    @property
    def battle_policy(self) -> BattlePolicy:
        return self._battle_policy


def main(args):
    n_agents = args.n_agents
    n_epochs = args.n_epochs
    n_vgc_epochs = args.n_epochs
    n_league_epochs = args.n_league_epochs
    base_port = args.base_port
    population_size = args.population_size
    surrogate_agent = [CompetitorManager(SurrogateCompetitor()) for _ in range(population_size)]
    move_roster = STANDARD_MOVE_ROSTER
    base_roster = RandomPkmRosterGenerator(None, n_moves_pkm=4, roster_size=100).gen_roster()
    constraints = VGCDesignConstraints(base_roster)
    evaluator = BaseMetaEvaluator()
    results = []
    for i in range(n_agents):
        address = ('localhost', base_port + i)
        conn = Client(address, authkey=f'Competitor {i}'.encode('utf-8'))
        competitor = ProxyCompetitor(conn)
        meta_data = StandardMetaData()
        meta_data.set_moves_and_pkm(base_roster, move_roster)
        gbe = GameBalanceEcosystem(evaluator, competitor, surrogate_agent, constraints, base_roster, meta_data,
                                   debug=True)
        gbe.run(n_epochs=n_epochs, n_vgc_epochs=n_vgc_epochs, n_league_epochs=n_league_epochs)
        results.append((competitor.name, gbe.total_score))
        conn.close()
    winner_name = ""
    max_score = 0.0
    for name, score in results:
        if score > max_score:
            winner_name = name
    print(winner_name + " wins the competition!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int, default=2)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_vgc_epochs', type=int, default=10)
    parser.add_argument('--n_league_epochs', type=int, default=10)
    parser.add_argument('--base_port', type=int, default=5000)
    parser.add_argument('--population_size', type=int, default=10)
    args = parser.parse_args()
    main(args)
