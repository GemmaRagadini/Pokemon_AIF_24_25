import random
from typing import Set, Tuple

from vgc.behaviour import TeamSelectionPolicy
from vgc.datatypes.Constants import DEFAULT_TEAM_SIZE
from vgc.datatypes.Objects import PkmFullTeam


class RandomTeamSelectionPolicy(TeamSelectionPolicy):

    def __init__(self, teams_size: int = DEFAULT_TEAM_SIZE, selection_size: int = DEFAULT_TEAM_SIZE):
        self.teams_size = teams_size
        self.selection_size = selection_size

    def get_action(self, d: Tuple[PkmFullTeam, PkmFullTeam]) -> Set[int]:
        """

        :param d: (self, opponent)
        :return: idx list of selected pokémon
        """
        ids = [i for i in range(self.teams_size)]
        random.shuffle(ids)
        return set(ids[:self.selection_size])


class FirstEditionTeamSelectionPolicy(TeamSelectionPolicy):

    def get_action(self, d: Tuple[PkmFullTeam, PkmFullTeam]) -> Set[int]:
        """
        Teams are selected as they are.

        :param d: (self, opponent)
        :return: idx list of selected pokémon
        """
        return {0, 1, 2}


class TerminalTeamSelection(TeamSelectionPolicy):
    """
    Terminal interface.
    """

    def get_action(self, s: Tuple[PkmFullTeam, PkmFullTeam]) -> Set[int]:
        print('~ Opponent Team ~')
        for p in s[1].pkm_list:
            print(p)
        print('~ My Team ~')
        for i, p in enumerate(s[0].pkm_list):
            print(i, '->', p)
        print('Select action in the format p p p with p in [0-5]')
        while True:
            valid = True
            try:
                t = input('Select Action: ')
                t = t.split()
                if len(t) != 3:
                    print('Invalid action. Select again.')
                    continue
                for m in t:
                    if not m.isdigit() and 0 < int(m) < len(s[0].pkm_list):
                        print('Invalid action. Select again.')
                        valid = False
                        break
                if valid:
                    break
            except:
                print('Invalid action. Select again.')
        print()
        return {int(t[0]), int(t[1]), int(t[2])}


# class GUITeamSelection TODO