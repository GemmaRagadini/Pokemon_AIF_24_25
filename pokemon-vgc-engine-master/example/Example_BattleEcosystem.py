import time as t
from Example_Competitor import ExampleCompetitor, MyCompetitor0, MyCompetitor1, MyCompetitor2, TerminalExampleCompetitor, MyCompetitor3, MyCompetitor4, MyCompetitor5, MyCompetitor6, MyCompetitor7
from vgc.balance.meta import StandardMetaData
from vgc.competition.Competitor import CompetitorManager
from vgc.ecosystem.BattleEcosystem import BattleEcosystem
from vgc.util.generator.PkmRosterGenerators import RandomPkmRosterGenerator
from vgc.util.generator.PkmTeamGenerators import RandomTeamFromRoster
import sys
import random

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


def Tournament():

    print("Let the Tournament begin")
    print( "stesso team")

    roster = RandomPkmRosterGenerator().gen_roster()
    meta_data = StandardMetaData()
    le = BattleEcosystem(meta_data, debug=True)
    # we start ny definig all the partecipant 
    team = RandomTeamFromRoster(roster).get_team()
    randomtrainer = CompetitorManager(MyCompetitor1("Player 1"))
    #randomtrainer.team = RandomTeamFromRoster(roster).get_team()
    randomtrainer.team = team

    Pokebob = CompetitorManager(MyCompetitor0("Player 0"))
    #Pokebob.team = RandomTeamFromRoster(roster).get_team()
    Pokebob.team = team

    minimax = CompetitorManager(MyCompetitor2("Player 2"))
    #minimax.team = RandomTeamFromRoster(roster).get_team()
    minimax.team = team

    minimax_killer= CompetitorManager(MyCompetitor3("Player 3"))
    #minimax_killer.team = RandomTeamFromRoster(roster).get_team()
    minimax_killer.team = team

    minimax_sorted_killer = CompetitorManager(MyCompetitor4("Player 4"))
    #minimax_sorted_killer.team = RandomTeamFromRoster(roster).get_team()
    minimax_sorted_killer.team = team

    minimax_sorted_killer_transpose = CompetitorManager(MyCompetitor5("Player 5"))
    #minimax_sorted_killer_transpose.team = RandomTeamFromRoster(roster).get_team()
    minimax_sorted_killer_transpose.team = team

    montecarlo_minimax = CompetitorManager(MyCompetitor6("Player 6"))
    #montecarlo_minimax.team = RandomTeamFromRoster(roster).get_team()
    montecarlo_minimax.team = team

    aggressive_montecarlo_minimax = CompetitorManager(MyCompetitor7("Player 7"))
    #aggressive_montecarlo_minimax.team = RandomTeamFromRoster(roster).get_team()
    aggressive_montecarlo_minimax.team = team


    trainers= [randomtrainer,Pokebob,minimax,minimax_killer,minimax_sorted_killer,minimax_sorted_killer_transpose,montecarlo_minimax,aggressive_montecarlo_minimax]
    
    #first 4 matches
    matches={'first':[], 'second':[] ,'thrid': [], 'fourth':[]}
    while len(trainers)>0:
        
        candidate = trainers.pop(random.randrange(len(trainers)))
        for i in matches.keys():
            if len(matches[i])<2:
                matches[i].append(candidate)
                break

    # play the first 4 matches and cadidate for the others
    matches['semi1'] = []
    matches['semi2'] = []
    matches['final'] = []
    winner=None

    for i in matches.keys():

        if i == 'semi1': #here we play the first semifinal
            print( f'the first semifinal between {matches[i][0].competitor.battle_policy.name} and {matches[i][1].competitor.battle_policy.name}')
            input()
            le.register(matches[i][0])
            le.register(matches[i][1])
            le.run(10)
            if le.win_counts[matches[i][0]] > le.win_counts[matches[i][1]]:
                print(f" {matches[i][0].competitor.battle_policy.name} wins the fist semifinal {le.win_counts[matches[i][0]]} games on 10 and has a win rate of {le.win_counts[matches[i][0]]/10}")
                input()
                le.unregister(matches[i][1])
                
                matches["final"].append(matches[i][0])
                le.unregister(matches[i][0])

            else :
                print(f" {matches[i][1].competitor.battle_policy.name} wins the first semifinal with {le.win_counts[matches[i][1]]} games on 10 and has a win rate of {le.win_counts[matches[i][1]]/10}")
                input()
                le.unregister(matches[i][0])
                matches["final"].append(matches[i][1])
                le.unregister(matches[i][1])

        elif i == 'semi2': #here we play the second semifinal
            print( f'the second semifinal between {matches[i][0].competitor.battle_policy.name} and {matches[i][1].competitor.battle_policy.name}')
            input()

            le.register(matches[i][0])
            le.register(matches[i][1])
            le.run(10)
            if le.win_counts[matches[i][0]] > le.win_counts[matches[i][1]]:
                print(f" {matches[i][0].competitor.battle_policy.name} wins the second semifinal {le.win_counts[matches[i][0]]} games on 10 and has a win rate of {le.win_counts[matches[i][0]]/10}")
                input()
                le.unregister(matches[i][1])
                
                matches["final"].append(matches[i][0])
                le.unregister(matches[i][0])

            else :
                print(f" {matches[i][1].competitor.battle_policy.name} wins the second semifinal with {le.win_counts[matches[i][1]]} games on 10 and has a win rate of {le.win_counts[matches[i][1]]/10}")
                input()
                le.unregister(matches[i][0])
                matches["final"].append(matches[i][1])
                le.unregister(matches[i][1])

        elif i == 'final': #here we play the final
            print( f'the final between {matches[i][0].competitor.battle_policy.name} and {matches[i][1].competitor.battle_policy.name}')
            input()

            le.register(matches[i][0])
            le.register(matches[i][1])
            le.run(10)
            if le.win_counts[matches[i][0]] > le.win_counts[matches[i][1]]:
                print(f" {matches[i][0].name} wins the final with {le.win_counts[matches[i][0]]} games on 10 and has a win rate of {le.win_counts[matches[i][0]]/10}")
                input()
                le.unregister(matches[i][1])
                winner = matches[i][0]
                le.unregister(matches[i][0])

            else :
                print(f" {matches[i][1].competitor.battle_policy.name} wins the final with {le.win_counts[matches[i][1]]} games on 10 and has a win rate of {le.win_counts[matches[i][1]]/10}")
                input()
                le.unregister(matches[i][0])
                winner = matches[i][1]
                le.unregister(matches[i][1])


        else: # here we play the normal matches
            print( f'the {i} match between {matches[i][0].competitor.battle_policy.name} and {matches[i][1].competitor.battle_policy.name}')
            input()
            le.register(matches[i][0])
            le.register(matches[i][1])
            le.run(10)
            if le.win_counts[matches[i][0]] > le.win_counts[matches[i][1]]:
                print(f" {matches[i][0].competitor.battle_policy.name} wins {le.win_counts[matches[i][0]]} games on 10 and has a win rate of {le.win_counts[matches[i][0]]/10}")
                input()
                le.unregister(matches[i][1])
                if len(matches["semi1"])<2:
                    matches["semi1"].append(matches[i][0])
                    le.unregister(matches[i][0])
                else:
                    matches["semi2"].append(matches[i][0])
                    le.unregister(matches[i][0])

            else :
                print(f" {matches[i][1].competitor.battle_policy.name} wins {le.win_counts[matches[i][1]]} games on 10 and has a win rate of {le.win_counts[matches[i][1]]/10}")
                input()
                le.unregister(matches[i][0])
                if len(matches["semi1"]) <2:
                    matches["semi1"].append(matches[i][1])
                    le.unregister(matches[i][1])
                else:
                    matches["semi2"].append(matches[i][1])
                    le.unregister(matches[i][1])

    print(f'The winner is {winner.competitor.battle_policy.name}')
    






if __name__ == '__main__':
    turnament=sys.argv[-1]
    if turnament== 't':
        Tournament()
    else:
        main()
