import torch

from dqn import DQN
from feature_proc import get_square_features_params, get_square_features, get_features, get_features_params, get_action_rotated_north_head
from goose_agents import RLAgent
from kaggle_environments import make

env = make("hungry_geese", debug=True)
env.reset()

# get_state = get_square_features
# adjust_action = get_action_rotated_north_head
# get_params = get_square_features_params


get_state = get_features
adjust_action = None
get_params = get_features_params


device = torch.device("cpu")
net = DQN(*get_params(env.configuration),
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 outputs=4,
                 device=device).to(device)

net.load_state_dict(torch.load("./results/5x5_DQN.net"))
net.eval()
players = [RLAgent(net, get_state, device, 0, adjust_action),
         RLAgent(net, get_state, device, 0, adjust_action),
         RLAgent(net, get_state, device, 0, adjust_action)]

env.run([RLAgent(net, get_state,  device, 0, adjust_action)] + players)
out = env.render(mode="html", width=500, height=400)

f = open("game.html", "w")
f.write(out)
f.close()
