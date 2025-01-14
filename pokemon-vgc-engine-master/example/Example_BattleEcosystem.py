import time
from Example_Competitor import ExampleCompetitor, MyCompetitor0, MyCompetitor1, TerminalExampleCompetitor
from vgc.balance.meta import StandardMetaData
from vgc.competition.Competitor import CompetitorManager
from vgc.ecosystem.BattleEcosystem import BattleEcosystem
from vgc.util.generator.PkmRosterGenerators import RandomPkmRosterGenerator
from vgc.util.generator.PkmTeamGenerators import RandomTeamFromRoster

N_PLAYERS = 2


def main():
    roster = RandomPkmRosterGenerator().gen_roster()
    meta_data = StandardMetaData()
    le = BattleEcosystem(meta_data, debug=True)
    cm1 = CompetitorManager(MyCompetitor0("Player 0"))
    cm1.team = RandomTeamFromRoster(roster).get_team()
    le.register(cm1)
    cm2 = CompetitorManager(MyCompetitor1("Player 1"))
    cm2.team = RandomTeamFromRoster(roster).get_team()
    le.register(cm2)
    # for i in range(N_PLAYERS):
    #     cm = CompetitorManager(ExampleCompetitor("Player %d" % i))
    #     cm.team = RandomTeamFromRoster(roster).get_team()
    #     le.register(cm)
     # Misurazione del tempo di esecuzione

  
    start_time = time.time()
    le.run(5)
    end_time = time.time()
    
    # Stampa del tempo impiegato
    print(f"Tempo impiegato: {end_time - start_time:.2f} secondi")

if __name__ == '__main__':
    main()
