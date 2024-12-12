import sys
import time

import numpy as np

from vgc.balance.meta import StandardMetaData
from vgc.behaviour.TeamBuildPolicies import TerminalTeamBuilder
from vgc.util.generator.PkmRosterGenerators import RandomPkmRosterGenerator

np.set_printoptions(threshold=sys.maxsize)


def main():
    roster = RandomPkmRosterGenerator().gen_roster()
    team_builder = TerminalTeamBuilder()
    t = time.time()
    team_builder.set_roster(roster)
    print(time.time() - t)
    # print(team_builder.matchup_table)
    print(team_builder.get_action(StandardMetaData()))


if __name__ == '__main__':
    main()
