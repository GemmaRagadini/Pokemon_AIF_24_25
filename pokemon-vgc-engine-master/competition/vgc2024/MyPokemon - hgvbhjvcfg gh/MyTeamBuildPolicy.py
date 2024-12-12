from typing import List
import numpy as np

from vgc.balance.meta import MetaData
from vgc.behaviour import TeamBuildPolicy
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES
from vgc.datatypes.Objects import Pkm, PkmTemplate, PkmFullTeam, PkmRoster
from vgc.datatypes.Types import PkmType

from MyBattlePolicy import match_up_eval

class MyTeamBuilder(TeamBuildPolicy):
    def __init__(self):
        self.roster = None

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        self.roster = roster

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        # Select Pokémon based on type coverage and defensive match-ups
        team_candidates = self.select_team_candidates()
        team = self.build_team(team_candidates)
        return PkmFullTeam(team)

    def select_team_candidates(self) -> List[PkmTemplate]:
        type_coverage = {t: 0 for t in PkmType}
        for pkm in self.roster:
            for move in pkm.moves:
                type_coverage[move.type] += 1

        team_candidates = []
        while len(team_candidates) < 3:
            best_score = float('-inf')
            best_candidate = None
            for pkm in self.roster:
                if pkm not in team_candidates:
                    score = self.evaluate_candidate(pkm, team_candidates, type_coverage)
                    if score > best_score:
                        best_score = score
                        best_candidate = pkm
            team_candidates.append(best_candidate)

        return team_candidates

    def evaluate_candidate(self, candidate: PkmTemplate, team: List[PkmTemplate], type_coverage: dict) -> float:
        score = 0
        for move in candidate.moves:
            score += type_coverage[move.type]
        for other_pkm in team:
            score += match_up_eval(candidate.type, other_pkm.type, [move.type for move in other_pkm.moves])
        return score

    def build_team(self, team_candidates: List[PkmTemplate]) -> List[Pkm]:
        team: List[Pkm] = []
        for pt in team_candidates:
            moves: List[int] = np.random.choice(DEFAULT_PKM_N_MOVES, DEFAULT_PKM_N_MOVES, False)
            team.append(pt.gen_pkm(moves))
        return team