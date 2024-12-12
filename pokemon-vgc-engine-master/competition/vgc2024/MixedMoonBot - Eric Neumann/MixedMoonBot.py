####### INFORMATION ###########
# Teamteilnehmer: Eric Neumann & Emanuel Thon



# My teambuild

from typing import List
import numpy as np
from copy import deepcopy
from vgc.balance.meta import MetaData
from vgc.behaviour import TeamBuildPolicy, BattlePolicy
from vgc.behaviour.BattlePolicies import TypeSelector, estimate_damage
from vgc.competition.Competitor import Competitor
from vgc.datatypes.Constants import DEFAULT_N_ACTIONS, DEFAULT_PARTY_SIZE, DEFAULT_PKM_N_MOVES, TYPE_CHART_MULTIPLIER
from vgc.datatypes.Objects import GameState, Pkm, PkmFullTeam, PkmRoster, PkmTeam, PkmTemplate
from vgc.datatypes.Types import PkmStat
from vgc.engine.PkmBattleEnv import PkmBattleEnv

# My teambuild

class EATeamBuild(TeamBuildPolicy):
    """
    Benutze EA um ein Team zu erstellen, fange mit Populationsgröße mit Randomteams an
    Selection ist eine championship -> höchsten 16 von der Elo her kommen weiter (oder vielleicht nur 8?)
    von den 16 die weitergekommen sind, macht man mit 16*mutationRate Mutation (ein Pokemon tauschen)
    Restlichen bis zur Populationsgröße werden mit randoms aufgefüllt
    Nach jeder Runde die man überlebt, bekommt man +1 Score -> am Ende returne ich das Team mit höchstem Score
    """
    def __init__(self, generations: int = 50, populationRatio : int = 1, mutationRate : float = 1, goodPkmRoster = False):
        self.roster = None
        self.originRoster = None
        self.populationSize = 0
        self.winnerSize = 0
        self.generations = generations
        self.mutationRate = mutationRate
        self.populationRatio = populationRatio
        self.goodPkmRoster = goodPkmRoster
        #print("Hier bin ich")
    
    
    def set_roster(self, roster: PkmRoster):
        if(self.goodPkmRoster):
            self.roster = self.removeDoubleTypeFromRoster(roster)
        else:
            self.roster = roster

        self.populationSize = int(len(self.roster)*self.populationRatio)
        #print("Rostergröße: " + str(len(roster)))
        #print("Popsize: " + str(self.populationSize))
        self.winnerSize = int(self.populationSize/2)
    
    
    def set_roster(self, roster: PkmRoster, version: int = 0):
        self.originRoster = deepcopy(roster)
        if(self.goodPkmRoster):
            self.roster = self.removeDoubleTypeFromRoster(roster)
        else:
            self.roster = roster

        self.populationSize = int(len(self.roster)*self.populationRatio)
        #print("Rostergröße: " + str(len(roster)))
        #print("Popsize: " + str(self.populationSize))
        self.winnerSize = int(self.populationSize/4)
    
    def get_roster(self):
        return self.originRoster
        
    def run_teamBattles(self, t0, t1, agent0, agent1, n_battles):
        wins = [0,0]
        #print("Run team battle")
        env = PkmBattleEnv((PkmTeam(deepcopy(t0.pkm_list)), PkmTeam(deepcopy(t1.pkm_list))), encode=(agent0.requires_encode(), agent1.requires_encode()))
        #print("Setzte env auf")
        for _ in range(n_battles):
            s, _ = env.reset()
            t = False
            while not t:
                a0 = agent0.get_action(s[0])
                a1 = agent1.get_action(s[1])
                #Sometimes random attack, damit typeselector nicht festklemmt
                if(np.random.random() < 0.3):
                    s, _, t, _, _ = env.step([np.random.randint(0, DEFAULT_PKM_N_MOVES), np.random.randint(0, DEFAULT_PKM_N_MOVES)])
                else:
                    s, _, t, _, _ = env.step([a0, a1])
            wins[env.winner] += 1
        
        #print("Das returne ich; " + str(wins))
        return wins

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        roster = list(self.roster)
        #print("Hier bin ich1")
        #eigentlich muss ich generations mit Zeit austauschen, damit man das maximum rauholt
        population : List[self.EAIndividuum] = []
        for _ in range(self.populationSize):
            pre_selection: List[PkmTemplate] = np.random.choice(roster,3, False)
            team: List[Pkm] = []
            for pt in pre_selection:
                team.append(pt.gen_pkm([0, 1, 2, 3]))
            population.append(self.EAIndividuum(PkmFullTeam(team)))

        #print("Das hier ist die ganze Population;")
        #for indiv in population:
        #    print(indiv.team)
        #print("Hier sieht man die Anfangspopulation:")
        #for comp in population:
        #    print(comp.team)

        #print("Hier sieht man das Roster:")
        #for pkm in self.roster:
        #    print(pkm)
        #print("Hier bin ich2")
        for i in range(self.generations):
            #print("Startet generation: " + str(i))
            for indiv in population:
                #print("Hier bin ich3")
                indiv.fitness = 0
                enemies : List[self.EAIndividuum] = np.random.choice(population,3)
                #print("Hier bin ich4")
                for enemy in enemies:
                    #print("Hier bin ich5")
                    #print("Myteam: " + str(indiv.team))
                    #print("Histeam: " + str(enemy.team))
                    wins = self.run_teamBattles(indiv.team, enemy.team, TypeSelector(), TypeSelector(), 1)
                    #print("Hier bin ich6")
                    if(wins[0] > wins[1]):
                        indiv.fitness += 1
                        enemy.fitness -= 1
                    else:
                        indiv.fitness -= 1
                        enemy.fitness += 1
                    
                    #print("Setzte fitness")
                    #indiv.team.reset()
                    #print("Resettete team")

            #print("Fertig mit allen Indivs")
            #print("So sieht zum beispiel population[0] aus: " + str(population[0].team))
            sortedList = sorted(population, key=lambda x: x.fitness, reverse=True)
            #print("Sortiert")
            population = []
            #print("Das ist die winnersize: " + str(self.winnerSize))

            for j in range(self.winnerSize):
                population.append(sortedList[j])
                #print("Das ist ein Gewinner: " + str(sortedList[j].team) + " mit Score: " + str(sortedList[j].fitness))

                #print("Ich packe dieses Team in die List: " + str(sortedList[j].team))
                #print("Das ist die Fitness: " + str(population[j].fitness))
            
            mutationCount = int(self.mutationRate*len(population))
            #print("MutationCount: " + str(mutationCount))
            for j in range(mutationCount):
                if(len(population) < self.populationSize):
                    population.append(self.EAIndividuum(self.mutate(population[j].team, self.roster)))
                if(len(population) < self.populationSize):
                    population.append(self.EAIndividuum(self.mutate(population[j].team, self.roster)))
                if(len(population) < self.populationSize):
                    population.append(self.EAIndividuum(self.mutate(population[j].team, self.roster)))
            #print("Mutation over")
            #print("Restgröße: " + str(self.populationSize - len(population)))
            for j in range(self.populationSize - len(population)):
                pre_selection: List[PkmTemplate] = np.random.choice(roster,3, False)
                team: List[Pkm] = []
                for pt in pre_selection:
                    team.append(pt.gen_pkm([0, 1, 2, 3]))
                population.append(self.EAIndividuum(PkmFullTeam(team)))
            #print("Added randoms")
            #print("Aktuelle Population:")
            #for comp in population:
            #    print("Dieser Champ mit dem Team " + str(comp.team) + " hat den Score: " + str(comp.score))
        sortedEloList = sorted(population, key=lambda x: x.fitness, reverse=True)
        #print("Hier ist der Winner")
        #print(sortedEloList[0].team)
        #print("Mit dem Score: " + str(sortedEloList[0].score))
        return sortedEloList[0].team
    
    def mutate(self, origin_team : PkmFullTeam, roster) -> List[PkmTemplate]:
        team = deepcopy(origin_team)
        newPkmRoster = []
        #print("Mutation called")
        for pt in roster:
            appendPkm = True
            for teamMember in team.pkm_list:
                if(teamMember.type == pt.type):
                    appendPkm = False
            if(appendPkm == True):
                newPkmRoster.append(pt.gen_pkm([0, 1, 2, 3]))
        #print("Hier komme ich auch noch hin")
        #print("TeamPkmlist länge = " + str(len(team.pkm_list)))
        #print("Rosterlänge: " + str(len(newPkmRoster)))
        index = np.random.randint(0, len(team.pkm_list))
        team.pkm_list[index] = newPkmRoster[np.random.randint(0, len(newPkmRoster))]
        #print("Neues Team: " + str(team))
        return team  

    def removeDoubleTypeFromRoster(self, roster : PkmRoster) -> PkmRoster:

        goodRoster = deepcopy(roster)

        for pt in roster:
            badPkm = True

            for i in range(len(pt.moves)):
                if(pt.moves[i].type == pt.type):
                    badPkm = False


            for i in range(len(pt.moves)):
                for j in range(len(pt.moves)):
                    if(i != j):
                        if(pt.moves[i].type == pt.moves[j].type):
                            badPkm = True
            if(badPkm == True):
                goodRoster.remove(pt)                

        return goodRoster
    
    class EAIndividuum():
        def __init__(self, team : PkmFullTeam):
            self.team = team
            self.fitness = 0

# my battle policy

class FlatMCTS(BattlePolicy):
    """
    Tree search algorithm that deals with adversarial paradigms by assuming the opponent acts in their best interest.
    Each node in this tree represents the worst case scenario that would occur if the player had chosen a specific
    choice.
    Source: http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf
    """

    def __init__(self, teamPolicy ,simulations = 100, withRandom = True):
        self.simulations = simulations
        self.teamBuildPolicy = teamPolicy
        self.withRandom = withRandom

    def get_action(self, g) -> int:  # g: PkmBattleEnv
        #startTime = time.time()

        roster = self.teamBuildPolicy.get_roster()
        guessing = True
        #guessing the moves of the opponent team
        
        opp_team : PkmTeam = g.teams[1]

        #print("Unrevealed Opp_Team: " + str(opp_team))
        myguess : PkmTeam = deepcopy(opp_team)

        counter = 0
        for pkm in opp_team.get_pkm_list():
            possiblePkms = deepcopy(roster)
            for pt in roster:
                if(pt.type != pkm.type):
                    possiblePkms.remove(pt)
            
            sameTypeRoster = possiblePkms

                #if(pt.max_hp < pkm.hp):          
                #    remove = True
            for pt in sameTypeRoster:
                remove = False
                for move in pkm.moves:
                    if(move.name != None):
                        if(move not in pt.moves):
                            #print("Entferne einen move")
                            remove = True
                if(remove == True):
                    possiblePkms.remove(pt)

            if(len(possiblePkms) == 0):
                #print("Hier geht was schief, benutze sameTypeGuessing")
                guessing = False
                #print("Und mein pkm: " + str(pkm))
                #print("Und das gegnerische Team: " + str(opp_team))
                #print("Und die gegnerische pkm liste: " + str(opp_team.get_pkm_list()))

        
            if(guessing == True):
                pkm_guess = possiblePkms[0]
            else:
                pkm_guess = sameTypeRoster[0]
            
            #print("Change moves")
            myguess.get_pkm_list()[counter].moves = pkm_guess.moves
            #print("Change moves done")
            counter += 1
        
        #print("Hier ist das gegnerische Team: " + str(opp_team))
        
        #print("Revealed Team: " + str(myguess))
                
        #print("Current Gamestate")
        #print("Team 0: " + str(g.teams[0]))
        #print("Team 1: " + str(g.teams[1]))
        #print("So sieht Myguess aus: " + str(myguess))
        g.set_predictions(myguess, g.teams[0])
        #g.set_predictions(g.teams[0], myguess)

        #print("Zeit zum Roster auswerten: " + str(time.time() - startTime))


        #startTime = time.time()




        #get my team
        my_team = g.teams[0]
        my_active = my_team.active
        my_attack_stage = my_team.stage[PkmStat.ATTACK]

        # get opp team
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_active_type = opp_active.type
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        bestActionIndex = -1
        bestNumberOfWins = -1
        for i in range(DEFAULT_N_ACTIONS):
            #PRUNING
            #PRUNING MOVES
            if(i < DEFAULT_PKM_N_MOVES):
                if(TYPE_CHART_MULTIPLIER[my_active.moves[i].type][opp_active_type] < 1.0 and my_active.moves[i].power > 0.0):
                    continue
                    #Man macht uneffektiven Move

            #PRUNING SWITCH ACTIONS
            if(i >= DEFAULT_PKM_N_MOVES):
                p = i - DEFAULT_N_ACTIONS
                for move in opp_active.moves:
                    if move.power > 0.0 and TYPE_CHART_MULTIPLIER[move.type][my_team.party[p].type] > 1.0:
                        continue
                        #Man switcht zu einem dummen Matchup




            numberOfWins = 0
            
            for _ in range(self.simulations):
                """
                currentGameState = deepcopy(g)

                #First Step always with move i

                o: GameState = deepcopy(currentGameState)
                o.teams = (o.teams[1], o.teams[0])
                opponentAction = GreedyBattle().get_action(o)

                s, _, _, _, _ = currentGameState.step([i, opponentAction]) 
                currentGameState = s[0]
                
                while(getBetterTeam(currentGameState, g) == -1 and getWinner(currentGameState) == -1):
                    o: GameState = deepcopy(currentGameState)
                    o.teams = (o.teams[1], o.teams[0])
                    opponentAction = GreedyBattle().get_action(o)

                    m: GameState = deepcopy(currentGameState)
                    m.teams = (m.teams[0], m.teams[1])
                    myaction = GreedyBattle().get_action(m)

                    s, _, _, _, _ = currentGameState.step([myaction, opponentAction]) 
                    currentGameState = s[0]

                if(getWinner(currentGameState) == 0):
                    #print("I won the sim")
                    numberOfWins+=1
                elif(getBetterTeam(currentGameState, g) == 0):
                    #print("I won the sim")
                    numberOfWins+=1
                """
                t = False
                currentGameState = deepcopy(g)

                #First Step always with move i
                o: GameState = deepcopy(currentGameState)
                o.teams = (o.teams[1], o.teams[0])
                opponentAction = self.get_greedy_action(o)

                s, _, t, _, _ = currentGameState.step([i, opponentAction]) 
                currentGameState = s[0]
                
                while not t :
                    o: GameState = deepcopy(currentGameState)
                    o.teams = (o.teams[1], o.teams[0])
                    opponentAction = self.get_greedy_action(o)

                    m: GameState = deepcopy(currentGameState)
                    m.teams = (m.teams[0], m.teams[1])
                    myaction =  self.get_greedy_action(m)

                    s, _, t, _, _ = currentGameState.step([myaction, opponentAction]) 
                    currentGameState = s[0]

                if(self.getWinner(currentGameState) == 0):
                    #print("I won the sim")
                    numberOfWins+=1
                


            #print("Bei der Aktion: " + str(i) + " habe ich soviele Wins gemacht: " + str(numberOfWins))
            if(numberOfWins > bestNumberOfWins):
                bestNumberOfWins = numberOfWins
                bestActionIndex = i


        #In case there is no good possible move
        if(bestNumberOfWins < self.simulations/2):
            weather = g.weather.condition

            # get most damaging move from my active pkm
            damage: List[float] = []
            for move in my_active.moves:
                damage.append(estimate_damage(move.type, my_active.type, move.power, opp_active_type,
                                            my_attack_stage, opp_defense_stage, weather) * move.acc)
            #print("Mäßige Aktion")
            return int(np.argmax(damage))  # use active pkm best damaging move
        
        #print("Simualtions: " + str(self.simulations) + ", Zeit: " + str(time.time() - startTime) + " und ich glaube zu gewinnen: " + str(bestNumberOfWins>self.simulations))
        #print("Mache gute Aktion")
        return bestActionIndex

    def get_greedy_action(self, g: GameState):
        # get weather condition
        weather = g.weather.condition


        # get my pkms
        my_team = g.teams[0]
        my_active = my_team.active
        my_attack_stage = my_team.stage[PkmStat.ATTACK]

        # get my pkmssorted(roster, key=lambda x: x.max_hp, reverse=True)
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_active_type = opp_active.type
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        # get most damaging move from my active pkm
        damage: List[float] = []
        for move in my_active.moves:
            damage.append(estimate_damage(move.type, my_active.type, move.power, opp_active_type,
                                            my_attack_stage, opp_defense_stage, weather) * move.acc)
        
        return int(np.argmax(damage))  # use active pkm best damaging move


    def getWinner(self, gameState) -> int:
        t0 = gameState.teams[0]
        t1 = gameState.teams[1]

        if(self.n_fainted(t0) == DEFAULT_PARTY_SIZE+1):
            return 1
        if(self.n_fainted(t1) == DEFAULT_PARTY_SIZE+1):
            return 0
        return -1

    def n_fainted(self, t: PkmTeam):
        fainted = 0
        fainted += t.active.hp == 0
        if len(t.party) > 0:
            fainted += t.party[0].hp == 0
        if len(t.party) > 1:
            fainted += t.party[1].hp == 0
        return fainted


class MixedMoonCompetitor(Competitor):

    def __init__(self, name: str = "MixedMoon"):
        self._name = name
        self.goodPkmRoster = True
        self.withRandom = False
        self.my_team_build_policy = EATeamBuild(50, 0.4, 1, self.goodPkmRoster)
        self.my_battle_policy = FlatMCTS(self.my_team_build_policy, 2, self.withRandom)
        #self.my_team_build_policy = RandomTeamBuilder()


    @property
    def battle_policy(self) -> BattlePolicy:
        return self.my_battle_policy

    @property
    def team_build_policy(self) -> TeamBuildPolicy:
        return self.my_team_build_policy

    @property
    def name(self) -> str:
        return self._name