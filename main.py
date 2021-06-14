from replay_memory import Transition, ReplayMemory
from dqn import DQN
from rl_agent import RLAgent
from feature_proc import get_features, get_features_params

import itertools
from random import shuffle
from tqdm import tqdm
from kaggle_environments import make, evaluate
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim


env = make("hungry_geese", debug=False)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
MEMORY_THRESHOLD = 10000
N_EPISODES_TRAIN = 10
N_ACTIONS = 4

policy_net = DQN(*get_features_params(env.configuration),
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 outputs=N_ACTIONS,
                 device=device).to(device)

target_net = DQN(*get_features_params(env.configuration),
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 outputs=N_ACTIONS,
                 device=device).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(MEMORY_THRESHOLD)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(torch.stack(batch.state, dim=0, out=None)).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.squeeze(1)

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def augment(batch):
    # random horizontal flip
    flip_mask = np.random.rand(len(batch['states'])) < 0.5
    batch['states'][flip_mask] = batch['states'][flip_mask].flip(-1)
    batch['actions'][flip_mask] = torch.where(batch['actions'][flip_mask] > 0, 4 - batch['actions'][flip_mask], 0) # 1 -> 3, 3 -> 1

    # random vertical flip (and also diagonal)
    flip_mask = np.random.rand(len(batch['states'])) < 0.5
    batch['states'][flip_mask] = batch['states'][flip_mask].flip(-2)
    batch['actions'][flip_mask] = torch.where(batch['actions'][flip_mask] < 3, 2 - batch['actions'][flip_mask], 3) # 0 -> 2, 2 -> 0

    # shuffle opponents channels
    permuted_axs = list(itertools.permutations([0, 1, 2]))
    permutations = [torch.tensor(permuted_axs[i]) for i in np.random.randint(6, size=len(batch['states']))]
    for i, p in enumerate(permutations):
        shuffled_channels = torch.zeros(3, batch['states'].shape[2], batch['states'].shape[3])
        shuffled_channels[p] = batch['states'][i, 1:4]
        batch['states'][:, 1:4] = shuffled_channels
    return batch


def collect_data(player, players):
    shuffle(players)
    trainer = env.train([None] + players)
    observation = trainer.reset()
    prev_obs = observation
    done = False
    prev_heads = [None for _ in range(4)]
    prev_state = get_features(observation, env.configuration, prev_heads)
    # start rollout
    while not done:
        # cache previous state
        for i, g in enumerate(observation['geese']):
            if len(g) > 0:
                prev_heads[i] = prev_obs['geese'][i][0]
        # make a move
        action = player.select_action(prev_state)
        # observe
        observation, reward, done, _ = trainer.step(['NORTH', 'EAST', 'SOUTH', 'WEST'][action])
        if not done:
            state = get_features(observation, env.configuration, prev_heads)
        else:
            state = None
        reward = torch.tensor([[reward]], device=device, dtype=torch.long)
        memory.push(prev_state, action, state, reward)

        prev_obs = observation
        prev_state = state


def train(n_episodes=N_EPISODES_TRAIN):
    losses_hist = {'clip': [], 'value-1': [], 'value-2': [], 'ent': [], 'lr': []}
    win_rates = {'Score': [], 'Rank': []}
    gammas = {'1': 0.8, '2': 0.8}
    lambdas = {'1': 0.7, '2': 0.7}
    print('-Start Training')
    for episode in tqdm(range(n_episodes), desc='Episode', leave=False):
        # update statistics
        target_net.eval()
        player = RLAgent(target_net, device)
        players = ['greedy'] * 3
        scores = evaluate("hungry_geese", [player] + players, num_episodes=30, debug=False)
        win_rates['Score'].append(np.mean([r[0] for r in scores]))
        win_rates['Rank'].append(np.mean([sum(r[0] <= r_ for r_ in r if r_ is not None) for r in scores]))
        # collect data
        collect_data(player, players)
        # perform training
        optimize_model()
        # Update the target network, copying all weights and biases in DQN
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return win_rates


def print_win_rates(win_rates):
    t = np.arange(len(win_rates['Score']))
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Score', color=color)
    ax1.plot(t, win_rates['Score'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Rank', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, win_rates['Rank'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Performance of DQN-net vs 3 Greedy Agents')
    plt.show()


def main():
    res = train()
    print_win_rates(res)
    # env.run([None, "greedy", "greedy", "greedy"])
    # env.render(mode="html", width=500, height=450)


if __name__ == '__main__':
    main()
