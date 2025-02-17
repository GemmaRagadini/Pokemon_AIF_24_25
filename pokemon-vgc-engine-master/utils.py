import numpy as np
import time as t
import sys
sys.path.append('./example')
from example import Example_BattleEcosystem, Example_Competitor
from vgc.behaviour import *
from vgc.datatypes.Constants import *
from vgc.datatypes.Objects import *
from vgc.datatypes.Types import * 
from copy import deepcopy
from typing import List
from vgc.balance.meta import StandardMetaData
from vgc.competition.Competitor import CompetitorManager
from vgc.ecosystem.BattleEcosystem import BattleEcosystem
from vgc.util.generator.PkmRosterGenerators import RandomPkmRosterGenerator
from vgc.util.generator.PkmTeamGenerators import RandomTeamFromRoster
from vgc.competition.Competitor import Competitor

N_PLAYERS = 2
player_classes = [Example_Competitor.MyCompetitor0, Example_Competitor.MyCompetitor1, Example_Competitor.MyCompetitor2, Example_Competitor.MyCompetitor3,
                   Example_Competitor.MyCompetitor4, Example_Competitor.MyCompetitor5]

def main():
    roster = RandomPkmRosterGenerator().gen_roster()
    meta_data = StandardMetaData()
    le = BattleEcosystem(meta_data, debug=True)
    n_epochs = 10
    times1, wins_0_different,rate1, policy_name1 = SingleCombat(n_epochs,le,roster, player_classes[0], player_classes[1])
    print(f"Player 0 con {policy_name1} ha vinto: {wins_0_different} partite su {n_epochs}, win rate {rate1}, tempo impiegato {times1}")

def SingleCombat(n_epochs, le:BattleEcosystem, roster, *players):    
    wins_player0 = 0
    wins_player1 = 0
    start_time = t.time()
    for i in range(n_epochs): 
        print(f"Epoch {i} of {n_epochs}")
        cm1 = CompetitorManager(players[0]("Player 0"))
        cm1.team = RandomTeamFromRoster(roster).get_team()
        le.register(cm1)
        cm2 = CompetitorManager(players[1]("Player 1"))
        cm2.team = RandomTeamFromRoster(roster).get_team()
        le.register(cm2)
        # single epoch
        le.run(1)
        # counts wins of Player 0 and Player 1
        wins_player0 += le.win_counts[cm1]
        wins_player1 += le.win_counts[cm2]        
        le.unregister(cm1)
        le.unregister(cm2)
    end_time = t.time()
    time = end_time-start_time
    return time,wins_player0,(wins_player0/n_epochs)*100, cm1.competitor.battle_policy.name

def Tournament():
    print("Let the Tournament begin")
    roster = RandomPkmRosterGenerator().gen_roster()
    meta_data = StandardMetaData()
    # trainers 
    Pokebob = CompetitorManager(Example_Competitor.MyCompetitor0("Player 0"))
    randomtrainer = CompetitorManager(Example_Competitor.MyCompetitor1("Player 1"))
    minimax = CompetitorManager(Example_Competitor.MyCompetitor2("Player 2"))
    minimax_killer= CompetitorManager(Example_Competitor.MyCompetitor3("Player 3"))
    minimax_my_eval = CompetitorManager(Example_Competitor.MyCompetitor4("Player 4"))
    minimax_killer_my_eval = CompetitorManager(Example_Competitor.MyCompetitor5("Player 5"))
    trainers= [Pokebob, randomtrainer, minimax,minimax_killer, minimax_my_eval, minimax_killer_my_eval] 
    scores = [0] * len(trainers) 
    n_epochs = 10 # number epochs for each match
    # italian tournament with each trainer having a randomly generated team for each battle
    print("Tournament")
    for i in range(len(trainers)):
        for j in range(i + 1, len(trainers)):  # Avoid duplicate matches
            print(f"Begin Match between {trainers[i].competitor.battle_policy.name} and {trainers[j].competitor.battle_policy.name}")
            wins_player0 = 0
            wins_player1 = 0
            for k in range(n_epochs):
                print(f"Epoch {k} of {n_epochs}")
                try:
                    le = BattleEcosystem(meta_data, debug=True)
                    trainers[i].team = RandomTeamFromRoster(roster).get_team()
                    trainers[j].team = RandomTeamFromRoster(roster).get_team()
                    le.register(trainers[i])
                    le.register(trainers[j])
                    # match
                    le.run(1)
                    wins_player0 += le.win_counts[trainers[i]]
                    wins_player1 += le.win_counts[trainers[j]]   
                    le.unregister(trainers[i])
                    le.unregister(trainers[j])
                except Exception as e:
                    import traceback
                    traceback.print_exc()
            scores[i] += wins_player0
            scores[j] += wins_player1
            print(f"Match Results: {trainers[i].competitor.battle_policy.name} won {wins_player0}, {trainers[j].competitor.battle_policy.name} won {wins_player1}")
    # sorted ranking
    ranking = sorted(zip(trainers, scores), key=lambda x: x[1], reverse=True)
    print("\nRanking tournament:\n")
    for i, (trainer, score) in enumerate(ranking, start=1):
        print(f"{i}. {trainer.competitor.battle_policy.name} - {score} punti")
