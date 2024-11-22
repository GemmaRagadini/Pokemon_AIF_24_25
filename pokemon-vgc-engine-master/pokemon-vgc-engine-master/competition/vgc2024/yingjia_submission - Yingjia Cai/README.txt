My name is Yingjia Cai, and my Discord ID is stella_vte.

Submission Contents:
My_Competitor.py
Adapted from Example_Competitor.py.

My_RemoteCompetitor.py
Adapted from Example_RemoteCompetitor.py. It Listens to port 5000 + <agent_id>.

Battle_agent_eval.py
My agent, that loads model parameters from "model_random_team_self_play.pth", use get_action to choose action.

model_random_team_self_play.pth
Contains the model parameters used in the DQN network.

My_encoder.py
my_encode_state is a custom function for encoding the game state an additional feature, damage estimation.


Directory Structure:
All files are located in the same directory.

Notes:
I've iterated several times on my agents, but none have yet outperformed the damage-focused agent, which has been unexpected.

Additional Information:
If you're interested in details about my model, including the training process, feel free to reach out to me.  I'm particularly curious about the performance of other participants' DQN-based agents in this program.