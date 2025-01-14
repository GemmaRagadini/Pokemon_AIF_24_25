from enum import Enum
from random import shuffle
from typing import List, Tuple

from vgc.balance.meta import MetaData
from vgc.competition.BattleMatch import BattleMatch
from vgc.competition.Competitor import CompetitorManager
from vgc.competition.Elo import elo_rating
from vgc.datatypes.Constants import DEFAULT_MATCH_N_BATTLES


class Strategy(Enum):
    RANDOM_PAIRING = 0
    ELO_PAIRING = 1


class BattleEcosystem:

    def __init__(self, meta_data: MetaData, debug=False, render=False, n_battles=DEFAULT_MATCH_N_BATTLES,
                 pairings_strategy: Strategy = Strategy.RANDOM_PAIRING, update_meta=False):
        self.meta_data = meta_data
        self.competitors: List[CompetitorManager] = []
        self.debug = debug
        self.render = render
        self.n_battles = n_battles
        self.pairings_strategy = pairings_strategy
        self.update_meta = update_meta
        ##
        self.win_counts = {}  # Dizionario per tracciare le vittorie di ogni CompetitorManager

    def register(self, cm: CompetitorManager):
        if cm not in self.competitors:
            self.competitors.append(cm)
            self.win_counts[cm] = 0  # Inizializza il contatore delle vittorie per il nuovo CompetitorManager
            

    def unregister(self, cm: CompetitorManager):
        self.competitors.remove(cm)
        del self.win_counts[cm]  # Rimuove il CompetitorManager dal dizionario delle vittorie

    def run(self, n_epochs: int):
        epoch = 0
        while epoch < n_epochs:
            self.__run_matches(self.__schedule_matches())
            epoch += 1
        self.print_results()  # Stampa i risultati dopo tutti gli epoch
        
    def __schedule_matches(self) -> List[Tuple[CompetitorManager, CompetitorManager]]:
        n_matches = len(self.competitors) // 2
        matches: List[Tuple[CompetitorManager, CompetitorManager]] = []
        if self.pairings_strategy == Strategy.RANDOM_PAIRING:
            shuffle(self.competitors)
        elif self.pairings_strategy == Strategy.ELO_PAIRING:
            sorted(self.competitors, key=lambda x: x.elo)
        for i in range(n_matches):
            matches.append((self.competitors[2 * i], self.competitors[2 * i + 1]))
        return matches

    def __run_matches(self, pairs: List[Tuple[CompetitorManager, CompetitorManager]]):
        for pair in pairs:
            cm0, cm1 = pair
            match = BattleMatch(cm0, cm1, self.n_battles, self.debug, self.render, meta_data=self.meta_data,
                                update_meta=self.update_meta)
            match.run()
            winner = cm0 if match.winner() == 0 else cm1
            self.win_counts[winner] += 1  # Incrementa il contatore delle vittorie per il vincitore
            cm0.elo, cm1.elo = elo_rating(cm0.elo, cm1.elo, 1 if match.winner() == 0 else 0)


    def print_results(self):
        # Stampa i risultati di tutte le vittorie
        print("\nRisultati:")
        for cm, wins in self.win_counts.items():
            print(f"{cm.competitor.name} con algoritmo {cm.competitor.battle_policy.name} ha vinto {wins} partite.")
