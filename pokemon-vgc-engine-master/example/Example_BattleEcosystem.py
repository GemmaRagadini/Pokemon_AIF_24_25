import time as t
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
    n_epochs = 20
 
    times1, wins_0_different,rate1, policy_name1 =different_teams(n_epochs,le,roster)
    times2, wins_0_same, rate2, policy_name2 =same_team(n_epochs,le,roster)

    print(f"Player 0 con diverso team con {policy_name1} ha vinto: {wins_0_different} partite su {n_epochs}, win rate {rate1}, tempo impiegato {times1}")
    print(f"Player 1 stesso team con {policy_name2} ha vinto: {wins_0_same} partite su {n_epochs}, win rate {rate2}, tempo impiegato {times2}")


    #VERSIONE STESSO TEAM PER OGNI EPOCA 
    # cm1 = CompetitorManager(MyCompetitor0("Player 0"))
    # cm1.team = RandomTeamFromRoster(roster).get_team()
    # le.register(cm1)
    # cm2 = CompetitorManager(MyCompetitor1("Player 1"))
    # cm2.team = RandomTeamFromRoster(roster).get_team()
    # le.register(cm2)  
    # start_time = time.time()
    # le.run(n_epochs)
    # end_time = time.time()
    
    # # Stampa del tempo impiegato
    # print(f"Tempo impiegato: {end_time - start_time:.2f} secondi")

def different_teams(n_epochs,le:BattleEcosystem,roster):

    wins_player0 = 0
    wins_player1 = 0
    start_time = t.time()
    for i in range(n_epochs): 
        cm1 = CompetitorManager(MyCompetitor0("Player 0"))
        team = RandomTeamFromRoster(roster).get_team()
        # cm1.team = RandomTeamFromRoster(roster).get_team()
        cm1.team = team
        le.register(cm1)
        cm2 = CompetitorManager(MyCompetitor1("Player 1"))
        team2 = RandomTeamFromRoster(roster).get_team()
        cm2.team = team2
        # cm2.team = RandomTeamFromRoster(roster).get_team()
        le.register(cm2)
        # Esegui una singola epoca
        le.run(1)
        # Stampa le vittorie di Player 0 e Player 1
        wins_player0 += le.win_counts[cm1]
        wins_player1 += le.win_counts[cm2]        

        le.unregister(cm1)
        le.unregister(cm2)
    end_time = t.time()
    time = end_time - start_time

    return time,wins_player0,(wins_player0/n_epochs)*100,cm1.competitor.battle_policy.name


def same_team(n_epochs, le:BattleEcosystem,roster):
    
    wins_player0 = 0
    wins_player1 = 0
    start_time = t.time()

    for i in range(n_epochs): 
        cm1 = CompetitorManager(MyCompetitor0("Player 0"))
        cm1.team = RandomTeamFromRoster(roster).get_team()
        le.register(cm1)
        cm2 = CompetitorManager(MyCompetitor1("Player 1"))
        cm2.team = RandomTeamFromRoster(roster).get_team()
        le.register(cm2)
        # Esegui una singola epoca
        le.run(1)
        # Conta le vittorie di Player 0 e Player 1
        wins_player0 += le.win_counts[cm1]
        wins_player1 += le.win_counts[cm2]        
        le.unregister(cm1)
        le.unregister(cm2)
    end_time = t.time()
    time = end_time-start_time
  
    return time,wins_player0,(wins_player0/n_epochs)*100, cm1.competitor.battle_policy.name

if __name__ == '__main__':
    main()
