from collections import defaultdict
from itertools import combinations
import numpy as np
from vgc.balance.meta import MetaData
from vgc.behaviour import BattlePolicy, TeamSelectionPolicy, TeamBuildPolicy
from vgc.behaviour.BattlePolicies import Minimax, RandomPlayer, TerminalPlayer
from vgc.behaviour.TeamBuildPolicies import TerminalTeamBuilder, RandomTeamBuilder
from vgc.behaviour.TeamSelectionPolicies import FirstEditionTeamSelectionPolicy
from vgc.competition.Competitor import Competitor
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, TYPE_CHART_MULTIPLIER
from vgc.datatypes.Objects import Pkm, PkmFullTeam, PkmRoster, PkmTemplate, Weather
from vgc.datatypes.Types import N_STATS, PkmStatus, PkmStat
from Thunder_BattlePolicies import ThunderPlayer, get_dp_move_turn


def myprint(*args, **kargs):
    print(*args, **kargs)


class Info:
    def __init__(self, pkm: Pkm, pkms: list[Pkm]) -> None:
        self.pkm = pkm
        self.has_sekka = False
        for move in pkm.moves:
            if move.priority:
                self.has_sekka = True
                break
        self.has_bulk = False
        self.has_speed = False
        for move in pkm.moves:
            if move.name == "Bulk Up":
                self.has_bulk = True
            if (move.stat == PkmStat.SPEED and
                ((move.stage < 0 and move.target == 1)
                 or (move.stage > 0 and move.target == 0))):
                self.has_speed = True
        self.turn_max = -1
        self.offence_one_turn_cnt = 0
        self.deffence_one_turn_cnt = 0
        self.yuuri_cnt = 0
        self.power_pp = sum(
            [move.max_pp for move in pkm.moves if move.power > 0 or move.fixed_damage > 0])
        self.offence_turns = []
        self.deffence_turns = []
        for op_pkm in pkms:
            # if self.pkm is op_pkm:
            #     continue
            _, offence_best_turn,_ = get_dp_move_turn(Weather(), pkm, [
                0] * N_STATS, op_pkm, [0] * N_STATS)
            _, deffence_best_turn,_ = get_dp_move_turn(Weather(), op_pkm, [
                0] * N_STATS, pkm, [0] * N_STATS)

            self.offence_turns.append(offence_best_turn)
            self.deffence_turns.append(deffence_best_turn)

            self.turn_max = max(offence_best_turn, self.turn_max)
            if offence_best_turn == 1:
                self.offence_one_turn_cnt += 1
            if deffence_best_turn == 1:
                self.deffence_one_turn_cnt += 1
            if offence_best_turn < deffence_best_turn:
                self.yuuri_cnt += 1

        # myprint("turn_max", self.turn_max)
        # myprint("offence_one_turn_cnt", self.offence_one_turn_cnt)

        # raise Exception("Info normal")


class Infos:

    def __init__(self, infos: list[Info]):
        self.infos = infos
        len_types = len(TYPE_CHART_MULTIPLIER)
        nigate_type_map = defaultdict(int)
        for info in infos:
            for offense_type in range(len_types):
                if TYPE_CHART_MULTIPLIER[offense_type][info.pkm.type] == 2:
                    nigate_type_map[offense_type] += 1
        self.nigate_cnt = sum(
            [nigate_value-1 for nigate_value in nigate_type_map.values() if nigate_value >= 2])
        self.turn_max_max = sum([info.turn_max for info in infos])
        self.offence_one_turn_cnt_sum = sum(
            [info.offence_one_turn_cnt for info in infos])
        self.pp_sum = sum(
            [info.power_pp for info in infos])
        self.has_sekka_cnt = 0
        for info in infos:
            if info.has_sekka:
                self.has_sekka_cnt += 1
        self.has_bulk_cnt = 0
        for info in infos:
            if info.has_bulk:
                self.has_bulk_cnt += 1

        self.deffence_one_turn_cnt = 0
        for i in range(len(infos[0].deffence_turns)):
            deffence_turns = [info.deffence_turns[i] for info in infos]
            min_turn = min(deffence_turns)
            if min_turn == 1:
                self.deffence_one_turn_cnt += 1
        self.offence_one_turn_cnt = 0
        for i in range(len(infos[0].offence_turns)):
            offence_turns = [info.offence_turns[i] for info in infos]
            min_turn = min(offence_turns)
            if min_turn == 1:
                self.offence_one_turn_cnt += 1

        # self.turn_max_max = max([info.turn_max for info in infos])


class ThunderBuilder(TeamBuildPolicy):
    """
    Agent that selects teams randomly.
    """

    def __init__(self):
        self.roster = None

    def set_roster(self, roster: PkmRoster, ver: int = 0):
        self.roster = roster

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        n_pkms = len(self.roster)
        pkms: list[Pkm] = [pt.gen_pkm([0, 1, 2, 3]) for pt in self.roster]
        # pkms: list[Pkm] = [pt.gen_pkm([0, 1, 2, 3])
        #                    for pt in self.roster if pt.max_hp >= 230]
        infos = [Info(pkm, pkms) for pkm in pkms]

        # infos = sorted(infos, key=lambda x: (
        #     x.pkm.hp, -x.turn_max, x.has_speed, x.pkm.hp, x.offence_one_turn_cnt), reverse=True)

        # infos = sorted(infos, key=lambda x: (
        #     -x.deffence_one_turn_cnt, x.has_sekka, x.pkm.hp, -x.turn_max, x.has_speed, x.pkm.hp, x.offence_one_turn_cnt), reverse=True)

        # infos = sorted(infos, key=lambda x: (
        #     x.yuuri_cnt, x.has_sekka), reverse=True)

        infos = sorted(infos, key=lambda x: (
            x.pkm.hp >= 230, x.yuuri_cnt, x.has_sekka), reverse=True)

        # infos = sorted(infos, key=lambda x: (
        #     x.pkm.hp >= 230, x.yuuri_cnt), reverse=True)

        return PkmFullTeam([info.pkm for info in infos[:3]])

        type_best_info_dict = dict()
        for info in infos:
            if info.pkm.type not in type_best_info_dict:
                type_best_info_dict[info.pkm.type] = info
        if len(type_best_info_dict) >= 3:
            infos = list(type_best_info_dict.values())
        max_hp = max([info.pkm.hp for info in infos])
        max_hp_infos = [info for info in infos if info.pkm.hp == max_hp]
        if len(max_hp_infos) >= 3:
            infos = max_hp_infos

        infoses = [Infos(comb) for comb in combinations(infos, 3)]

        def key_f(x: Infos):
            return (-x.nigate_cnt, -x.deffence_one_turn_cnt)
            # return (-x.deffence_one_turn_cnt)
            # return (-x.deffence_one_turn_cnt, x.offence_one_turn_cnt, -x.turn_max_max, -x.nigate_cnt)
            # return (int(x.has_bulk_cnt > 0)*2 - x.deffence_one_turn_cnt, x.offence_one_turn_cnt, -x.turn_max_max, -x.nigate_cnt)
            # return (x.has_bulk_cnt > 0, -x.deffence_one_turn_cnt, x.offence_one_turn_cnt, -x.turn_max_max, -x.nigate_cnt)
        infoses = sorted(
            infoses, key=key_f, reverse=True)
        # infoses = sorted(
        #     infoses, key=lambda x: (-x.turn_max_max, -x.nigate_cnt, ), reverse=True)
        for infos in infoses:
            myprint(
                f"{infos.deffence_one_turn_cnt},{infos.offence_one_turn_cnt},{infos.turn_max_max:0.2},{infos.nigate_cnt},{infos.pp_sum}:", end="")
            for info in infos.infos:
                myprint(f"{info.pkm.type.name},[", end="")
                for move in info.pkm.moves:
                    myprint(f"{move.name},", end="")
                myprint(f"],", end="")
            myprint()
        # for comb in combinations(infos, 3):
        #     score = get_score_party(comb)
        #     myprint(f"{score}:", end="")
        #     for info in comb:
        #         myprint(f"{info.pkm.type.name},", end="")
        #     myprint("*"*40)
        # assert (False)
        # infos = infoses[0].infos
        infos = infoses[0].infos
        team: list[Pkm] = [info.pkm for info in infos]
        return PkmFullTeam(team)

        # infos = sorted(infos, key=lambda x: (
        #     x.pkm.hp, -x.turn_max, x.offence_one_turn_cnt, x.yuuri_cnt), reverse=True)
        for info in infos:
            myprint("*"*40)
            myprint("one_cnt,turn_max", info.pkm.hp, info.yuuri_cnt,
                    info.offence_one_turn_cnt, info.turn_max)
            myprint(info.pkm.type.name)
        team: list[Pkm] = [info.pkm for info in infos]
        return PkmFullTeam(team)


class ThunderCompetitor(Competitor):

    def __init__(self, name: str = "Thunder"):
        self._name = name
        self._battle_policy = ThunderPlayer()
        self._team_selection_policy = FirstEditionTeamSelectionPolicy()
        self._team_build_policy = ThunderBuilder()

    @ property
    def name(self):
        return self._name

    @ property
    def team_build_policy(self) -> TeamBuildPolicy:
        # myprint("call team_build_policy")
        return self._team_build_policy

    @ property
    def team_selection_policy(self) -> TeamSelectionPolicy:
        return self._team_selection_policy

    @ property
    def battle_policy(self) -> BattlePolicy:
        # myprint("call battle_policy")
        self._battle_policy.roster = self._team_build_policy.roster
        return self._battle_policy
