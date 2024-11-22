from typing import List
import numpy as np

from vgc.behaviour import BattlePolicy
from vgc.datatypes.Objects import GameState
from vgc.behaviour.BattlePolicies import estimate_damage, match_up_eval
from vgc.datatypes.Types import PkmStat


class CampiaoPolicy(BattlePolicy):
    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g: GameState) -> int:
        weather = g.weather.condition

        my_team = g.teams[0]
        my_active = my_team.active
        my_party = my_team.party
        my_attack_stage = my_team.stage[PkmStat.ATTACK]
        my_speed_stage = my_team.stage[PkmStat.SPEED]

        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_active_type = opp_active.type
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]
        opp_speed_stage = opp_team.stage[PkmStat.SPEED]

        damage: List[float] = []
        for move in my_active.moves:
            damage.append(estimate_damage(move.type, my_active.type, move.power, opp_active_type,
                                          my_attack_stage, opp_defense_stage, weather))
        move_id = int(np.argmax(damage))

        # if we put opp pkm below 90% max hp, do max dmg move
        if opp_active.hp - damage[move_id] <= opp_active.max_hp * 0.1:
            return move_id

        if my_speed_stage >= opp_speed_stage:
            if match_up_eval(my_active.type, opp_active.type, list(map(lambda m: m.type, opp_active.moves))) <= 1.0:
                return move_id

            # if match up is not favorable
            # if there is entry hazard, then don't change pkm and do max dmg move instead
            if my_team.entry_hazard[1] > 0:
                return move_id

            # try to change to better pkm match up
            match_up: List[float] = []
            not_fainted = my_team.get_not_fainted()

            # No more pkm alive other than active one :(
            if len(not_fainted) == 1:
                return move_id

            for i in not_fainted:
                match_up.append(
                    match_up_eval(my_party[i].type, opp_active.type, list(map(lambda m: m.type, opp_active.moves))))
            if len(match_up) == 0:
                return 4
            return int(np.argmin(match_up)) + 4

        # Scenario 2: we act second
        # If the match up was favorable, then opp might swap pkm
        # Assuming this case, check for most likely swap and estimate max dmg move
        match_up: List[float] = []
        for pkm in opp_team.party:
            match_up.append(
                match_up_eval(my_active.type, pkm.type, list(map(lambda m: m.type, pkm.moves)))
            )

        opp_best_pkm = opp_team.party[int(np.argmax(match_up))]
        damage: List[float] = []
        for move in my_active.moves:
            damage.append(estimate_damage(move.type, my_active.type, move.power, opp_best_pkm.type,
                                          my_attack_stage, opp_defense_stage, weather))
        move_id = int(np.argmax(damage))

        # if we put opp pkm below 90% max hp, do max dmg move
        if opp_best_pkm.hp - damage[move_id] <= opp_best_pkm.max_hp * 0.1:
            return move_id

        if match_up_eval(my_active.type, opp_best_pkm.type, list(map(lambda m: m.type, opp_best_pkm.moves))) <= 1.0:
            return move_id

        # if match up is not favorable
        # if there is entry hazard, then don't change pkm and do max dmg move instead
        if my_team.entry_hazard[1] > 0:
            return move_id

        # try to change to better pkm match up
        match_up: List[float] = []
        not_fainted = my_team.get_not_fainted()

        # No more pkm alive other than active one :(
        if len(not_fainted) == 1:
            return move_id

        for i in not_fainted:
            match_up.append(
                match_up_eval(my_party[i].type, opp_best_pkm.type, list(map(lambda m: m.type, opp_best_pkm.moves))))

        return int(np.argmin(match_up)) + 4
