import random
from typing import Set, Tuple, List
from vgc.behaviour import TeamSelectionPolicy
from vgc.datatypes.Constants import DEFAULT_TEAM_SIZE
from vgc.datatypes.Objects import PkmFullTeam, Pkm

class Real_Good_Selection(TeamSelectionPolicy):
    print("let's go Real_Good_Selection ")
    def __init__(self, teams_size: int = DEFAULT_TEAM_SIZE, selection_size: int = DEFAULT_TEAM_SIZE):
        self.teams_size = teams_size
        self.selection_size = selection_size

    def get_action(self, d: Tuple[PkmFullTeam, PkmFullTeam]) -> Set[int]:
        print("Selection _ get_action")
        my_team = d[0].pkm_list
        opponent_team = d[1].pkm_list

        print(f"Evaluating team selection:")
        print(f"My team: {[pkm.name for pkm in my_team]}")
        print(f"Opponent team: {[pkm.name for pkm in opponent_team]}")

        scores = self.calculate_scores(my_team, opponent_team)

        selected_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.selection_size]

        print(f"Calculated scores: {scores}")
        print(f"Selected indices: {selected_indices}")

        return set(selected_indices)

    def calculate_scores(self, my_team: List[Pkm], opponent_team: List[Pkm]) -> List[float]:
        print("Selection _ calculate_scores")
        scores = []
        for pkm in my_team:
            score = 0
            for opp_pkm in opponent_team:
                matchup_score = self.evaluate_matchup(pkm, opp_pkm)
                print(f"Matchup {pkm.name} vs {opp_pkm.name}: {matchup_score}")
                score += matchup_score
            scores.append(score)
        return scores

    def evaluate_matchup(self, pkm: Pkm, opp_pkm: Pkm) -> float:
        print("Selection _ evaluate_matchup")
        advantage = 1.0
        disadvantage = 1.0
        for move in opp_pkm.moves:
            type_multiplier = self.type_effectiveness(move.type, pkm.type)
            if type_multiplier > 1.0:
                disadvantage *= type_multiplier
            elif type_multiplier < 1.0:
                advantage *= 1 / type_multiplier
        return advantage / disadvantage

    def type_effectiveness(self, move_type, target_type) -> float:
        print("Selection _ type_effectiveness")
        type_chart = {
            'NORMAL': {'ROCK': 0.5, 'GHOST': 0.0, 'STEEL': 0.5},
            'FIRE': {'FIRE': 0.5, 'WATER': 0.5, 'GRASS': 2.0, 'ICE': 2.0, 'BUG': 2.0, 'ROCK': 0.5, 'DRAGON': 0.5, 'STEEL': 2.0},
        }
        if move_type in type_chart and target_type in type_chart[move_type]:
            effectiveness = type_chart[move_type][target_type]
            print(f"Effectiveness of {move_type} on {target_type}: {effectiveness}")
            return effectiveness
        return 1.0

# Example usage
if __name__ == "__main__":
    my_team = PkmFullTeam([Pkm(name='Pikachu'), Pkm(name='Bulbasaur'), Pkm(name='Charmander')])
    opponent_team = PkmFullTeam([Pkm(name='Squirtle'), Pkm(name='Geodude'), Pkm(name='Eevee')])

    selection_policy = Real_Good_Selection()
    selected_indices = selection_policy.get_action((my_team, opponent_team))
    print(f'Selected Pokémon indices: {selected_indices}')
