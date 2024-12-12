import torch
import torch.nn as nn

from My_encoder import my_encode_state
from vgc.behaviour import BattlePolicy

GAME_STATE_ENCODE_LEN = 1188


class deep_q_agent(BattlePolicy):
    def __init__(self, num_move=4, num_switch=2, state_space=279, model_dict="model_random_team_self_play.pth"):
        self.num_action = num_move + num_switch
        self.state_space = state_space
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.policy_net = nn.Sequential(
            nn.Linear(self.state_space, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_action)
        ).to(self.device)
        self.policy_net.load_state_dict(torch.load(model_dict))
        self.policy_net.eval()

    def get_action(self, g):
        state = my_encode_state(g)
        state = torch.tensor(state).to(self.device)
        q_val = self.policy_net(state)
        action = torch.argmax(q_val).item()
        return action

    def requires_encode(self):
        return False

    def close(self):
        pass




