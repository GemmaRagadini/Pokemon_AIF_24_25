
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER
from vgc.datatypes.Objects import GameState
from vgc.datatypes.Types import PkmType, WeatherCondition, PkmStat, N_TYPES, N_STATUS, N_WEATHER, N_ENTRY_HAZARD
import numpy as np


def one_hot(p, n):
    b = [0] * n
    b[p] = 1
    return b


def estimate_damage(move_type: PkmType, pkm_type: PkmType, move_power: float,
                    move_prob: float, opp_pkm_type: PkmType,
                    attack_stage: int, defense_stage: int,
                    weather: WeatherCondition) -> float:
    # The function is adapted from DominikBaziukCompetitor
    stab = 1.5 if move_type == pkm_type else 1.
    if (move_type == PkmType.WATER and weather == WeatherCondition.RAIN) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.SUNNY):
        weather = 1.5
    elif (move_type == PkmType.WATER and weather == WeatherCondition.SUNNY) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.RAIN):
        weather = .5
    else:
        weather = 1.
    stage_level = attack_stage - defense_stage
    stage = (stage_level + 2.) / 2 if stage_level >= 0. else 2. / (np.abs(stage_level) + 2.)
    damage = TYPE_CHART_MULTIPLIER[move_type][opp_pkm_type] * stab * weather * stage * move_power * move_prob
    return damage


def my_encode_state(g: GameState):
    # The function is adapted from DominikBaziukCompetitor
    state = []
    weather = g.weather.condition

    # get my pkms
    my_team = g.teams[0]
    my_pkms = [my_team.active] + my_team.party

    # get opp team
    opp_team = g.teams[1]
    opp_active = opp_team.active
    opp_active_type = opp_active.type
    opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

    for i in range(3):
        if i == 0:
            my_attack_stage = my_team.stage[PkmStat.ATTACK]
        else:
            my_attack_stage = 0
        pkm = my_pkms[i]

        state.append(pkm.hp)
        state += one_hot(pkm.type, N_TYPES)
        state += one_hot(pkm.status, N_STATUS)

        move_state = []
        for move in pkm.moves:
            move_state.append(move.pp)
            move_state.append(move.recover)
            move_state.append(move.stat.value)
            move_state.append(move.stage / 2)
            move_state += one_hot(move.weather, N_WEATHER)
            move_state += one_hot(move.hazard, N_ENTRY_HAZARD)
            if pkm.hp == 0.0:
                move_state.append(0.0)
            else:
                move_state.append(estimate_damage(move.type, pkm.type, move.power,
                                                  move.prob, opp_active_type,
                                                  my_attack_stage, opp_defense_stage,
                                                  weather))
        state.extend(move_state)

        pkm = opp_team.active
        state.append(pkm.hp)
        state += one_hot(pkm.type, N_TYPES)

    state = np.asarray(state, dtype=np.float32)
    return state

def my_encode_pkm_roster(pkm_roster):
    state = []
    for pkm in pkm_roster:
        state.append(pkm.hp)
        state += one_hot(pkm.type, N_TYPES)
    # 19 * 51
    state += [0] * (19 * 3)
    return np.asarray(state, dtype=np.float32)


if __name__ == "__main__":
    #roster = gen_pkm_roster()

    #temp = my_encode_pkm_roster(roster)
    #print(temp.shape)  # 1026
    pass