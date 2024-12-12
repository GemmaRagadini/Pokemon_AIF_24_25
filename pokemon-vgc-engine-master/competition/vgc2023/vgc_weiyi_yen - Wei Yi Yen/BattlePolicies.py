from copy import deepcopy
# weiwei
from queue import PriorityQueue

from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import DEFAULT_N_ACTIONS


# weiwei's
class AStarSearch(BattlePolicy):

    def __init__(self):
        self.root = AStarNode()
        self.node_queue = PriorityQueue()
        self.node_queue.put(self.root)

    def requires_encode(self) -> bool:
        return False

    def close(self):
        pass

    def get_action(self, g) -> int:
        self.root.g = g
        while not self.node_queue.empty():
            current_parent = self.node_queue.get()
            # Expand nodes of current parent
            for i in range(DEFAULT_N_ACTIONS):
                s, _, _, _, _ = current_parent.g.step([i, 99])  # Opponent selects an invalid switch action
                if s[0].teams[0].active.hp == 0:
                    continue
                elif s[0].teams[1].active.hp == 0:
                    a = i
                    while current_parent != self.root:
                        a = current_parent.a
                        current_parent = current_parent.parent
                    return a
                else:
                    node = AStarNode()
                    node.parent = current_parent
                    node.a = i
                    node.g = deepcopy(s[0])
                    node.h = self.heuristic(node.g)
                    node.f = node.parent.f + 1 + node.h
                    self.node_queue.put(node)
        return 0

    def heuristic(self, g) -> float:

        return 0.0


class AStarNode:

    def __init__(self):
        self.parent = None
        self.a = None
        self.g = None
        self.h = 0.0
        self.f = 0.0

    def __lt__(self, other):
        return self.f < other.f
