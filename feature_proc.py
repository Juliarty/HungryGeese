import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from kaggle_environments import make

FEATURE_CHANNELS = 12


def get_position_from_index(index, columns):
    row = index // columns
    col = index % columns
    return row, col


def get_index_from_position(row, col, columns):
    return row * columns + col


def find_new_head_position(head_row, head_col, action, rows, columns):
    if action == 0: # north
        new_row, new_col = (head_row + rows - 1) % rows, head_col
    elif action == 1: # east
        new_row, new_col = head_row, (head_col + 1) % columns
    elif action == 2: # south
        new_row, new_col = (head_row + 1) % rows, head_col
    else: # west
        new_row, new_col = head_row, (head_col + columns - 1) % columns
    return new_row, new_col


def shift_head(head_id, action, rows=7, columns=11):
    head_row, head_col = get_position_from_index(head_id, columns)
    new_row, new_col = find_new_head_position(head_row, head_col, action, rows, columns)
    new_head_id = get_index_from_position(new_row, new_col, columns)
    return new_head_id


def get_previous_head(ids, last_action, rows, columns):
    if len(ids) > 1:
        return ids[1]
    return shift_head(ids[0], (last_action + 2) % 4, rows, columns)


def ids2locations(ids, prev_head, step, rows, columns):
    state = np.zeros((4, rows * columns))
    if len(ids) == 0:
        return state
    state[0, ids[0]] = 1 # goose head
    if len(ids) > 1:
        state[1, ids[1:-1]] = 1 # goose body
        state[2, ids[-1]] = 1 # goose tail
    if step != 0:
        state[3, prev_head] = 1 # goose head one step before
    return state


def get_features(observation, config, prev_heads):
    rows, columns = config['rows'], config['columns']
    geese = observation['geese']
    index = observation['index']
    step = observation['step']
    # convert indices to locations
    locations = np.zeros((len(geese), 4, rows * columns))
    for i, g in enumerate(geese):
        locations[i] = ids2locations(g, prev_heads[i], step, rows, columns)
    if index != 0: # swap rows for player locations to be in first channel
        locations[[0, index]] = locations[[index, 0]]
    # put locations into features
    features = np.zeros((FEATURE_CHANNELS, rows * columns))
    for k in range(4):
        features[k] = np.sum(locations[k][:3], 0)
        features[k + 4] = np.sum(locations[:, k], 0)
    features[-4, observation['food']] = 1 # food channel
    features[-3, :] = (step % config['hunger_rate']) / config['hunger_rate'] # hunger danger channel
    features[-2, :] = step / config['episodeSteps'] # timesteps channel
    features[-1, :] = float((step + 1) % config['hunger_rate'] == 0) # hunger milestone indicator
    features = torch.Tensor(features).reshape(-1, rows, columns)
    # roll
    head_id = geese[index][0]
    head_row = head_id // columns
    head_col = head_id % columns
    features = torch.roll(features, ((rows // 2) - head_row, (columns // 2) - head_col), dims=(-2, -1))
    return features


def get_features_params(config):
    # Channels, Height, Width
    return FEATURE_CHANNELS, config['rows'], config['columns']


def plot_features(features):
    fig, axs = plt.subplots(3, 4, figsize=(20, 10))
    for i in range(3):
        for j in range(4):
            sns.heatmap(features[i * 4 + j], ax=axs[i, j], cmap='Blues',
                        vmin=0, vmax=1, linewidth=2, linecolor='black', cbar=False)


def get_example_features():
    observation = {}
    observation['step'] = 104
    observation['index'] = 0
    observation['geese'] = [[46, 47, 36, 37, 48, 59, 58, 69],
                            [5, 71, 72, 6, 7, 73, 62, 61, 50, 51, 52, 63, 64, 53, 54],
                            [12, 11, 21, 20, 19, 8, 74, 75, 76, 65, 55, 56, 67, 1],
                            [23, 22, 32, 31, 30, 29, 28, 17, 16, 27, 26, 15, 14, 13, 24]]
    observation['food'] = [45, 66]
    prev_heads = [47, 71, 11, 22]
    env = make("hungry_geese", debug=False)

    return get_features(observation, env.configuration, prev_heads)


if __name__ == '__main__':
    f = get_example_features()
    plot_features(f.cpu().detach().numpy())
