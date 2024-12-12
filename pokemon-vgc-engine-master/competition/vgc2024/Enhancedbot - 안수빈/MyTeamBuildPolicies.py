import random
from vgc.behaviour import TeamBuildPolicy
from vgc.datatypes.Objects import MetaData, PkmFullTeam

class SimpleTeamBuilder(TeamBuildPolicy):
    def __init__(self):
        self.roster = None

    def set_roster(self, roster):
        self.roster = roster

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        # 랜덤하게 6마리 포켓몬 선택
        team = random.sample(self.roster, 6)

        # 각 포켓몬에 랜덤하게 4개 기술 선택
        for pkm in team:
            moves = random.sample(pkm.move_templates, 4)
            pkm.moves = [move.id for move in moves]

        return PkmFullTeam(team)