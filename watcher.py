import torch

from qestimator import QEstimator
from feature_proc import get_square_features, get_features, get_action_rotated_north_head
from goose_agents import RLAgent, AgentFactory
from kaggle_environments import make
from feature_transform import SquareWorld7x7FeatureTransform

env = make("hungry_geese", debug=True)
env.reset()
device = torch.device("cpu")

# get_state = get_square_features
# adjust_action = get_action_rotated_north_head
# get_params = get_square_features_params

feature_transform_class = SquareWorld7x7FeatureTransform

net = QEstimator(c=feature_transform_class.FEATURES_CHANNELS_IN,
                 h=feature_transform_class.FEATURES_HEIGHT,
                 w=feature_transform_class.FEATURES_WIDTH,
                 outputs=4,
                 device=device).to(device)

net.load_state_dict(torch.load("./results/5x5_DQN.net"))
net.eval()

agent_factory = AgentFactory(net, RLAgent, feature_transform_class, device)
player = agent_factory.create(eps_greedy=0)
players = [agent_factory.create(eps_greedy=0) for _ in range(3)]

env.run([player] + players)
out = env.render(mode="html", width=500, height=400)

f = open("game.html", "w")
f.write(out)
f.close()



