import math
from  kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Action, Configuration, row_col
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from kaggle_environments import make

FEATURE_CHANNELS = 12


# все состояния так-то ялвяются результатом поворота, а действие должно быть применимо, к неизмененному миру
# опредялеяет сколько поворотов против часовой стрелки нужно сделать, чтобы смотреть на север
# и крутит в обратную сторону данное число раз action
def get_action_rotated_north_head(obs, prev_heads, action_in_rotated_world: Action, rows, columns):
    rotates_num = get_north_head_rot_num(obs, prev_heads, rows, columns)
    actual_action = action_in_rotated_world
    # вращаем по часовой, т.к. get_north_head_rot_num() крутит против
    for i in range(rotates_num):
        if actual_action == Action.NORTH:
            actual_action = Action.EAST
        elif actual_action == Action.EAST:
            actual_action = Action.SOUTH
        elif actual_action == Action.SOUTH:
            actual_action = Action.WEST
        elif actual_action == Action.WEST:
            actual_action = Action.NORTH
    return actual_action


def get_eps(start_value, end_value, step, episodes):
    return end_value + (start_value - end_value) * (episodes - step) / episodes


def get_reward(prev_obs, observation: Observation, configuration: Configuration, device):
    reward = 0
    index = observation.index
    if len(observation.geese[index]) == len(prev_obs.geese[index]) and len(observation.geese[index]) < 10:
        reward = -0.1
    elif len(observation.geese[index]) > len(prev_obs.geese[index]):     # ate sth
        reward = len(observation.geese[index]) * 1.5
    elif len(observation.geese[index]) == 0 and len(prev_obs.geese[index]) >= 2:  # died like a hero
        reward = -1 * get_eps(10, 1, observation.step, configuration.episode_steps)
    elif len(observation.geese[index]) == 0 and len(prev_obs.geese[index]) == 1:  # died like a rat
        reward = -100

    if len(observation.geese[index]) > 0:
        # check whether he lost his chance to become bigger
        head_y, head_x = row_col(observation.geese[index][0], configuration.columns)
        for food in observation.food:
            food_y, food_x = row_col(food, configuration.columns)
            dist_x = get_distance(food_x, head_x, configuration.columns)
            dist_y = get_distance(food_y, head_y, configuration.rows)
            if dist_y == 0 and dist_x == 1 or \
               dist_y == 1 and dist_x == 0:
                reward -= 2
                break

    return torch.tensor([[reward]], device=device, dtype=torch.long)


def get_square_features(observation: Observation, prev_head, configuration: Configuration):
    new_config = Configuration(configuration.copy())
    new_config['columns'] = new_config['rows']
    observation, prev_head = transform_to_rows_x_rows(observation, prev_head, configuration.rows, configuration.columns)
    observation, prev_head = set_head_to_the_north(observation, prev_head, new_config)

    prev_head = [prev_head[0], -1, -1, -1]

    return get_features(observation, prev_head, new_config)


def shifted_element_y_x(el_id, shift_x, shift_y, rows, columns):
    el_y, el_x = get_position_from_index(el_id, columns)
    new_el_y = (el_y - shift_y) % rows
    new_el_x = (el_x - shift_x) % columns

    if new_el_x < 0:
        new_el_x += columns

    if new_el_y < 0:
        new_el_y += rows

    return new_el_y, new_el_x


def get_distance(x1, x2, max_coord):
    return np.min([int(math.fabs(x1 - x2)), int(max_coord - math.fabs(x1 - x2))])


# нумерация всех id начинается с левого верхнего угла
# ось координат расположена там же,  т.е. центр будет с координатами (5, 3) при rows, columns = 7, 11
# необходимо отрезать все, что дальше центра больше чем на 3 окошка
def transform_to_central_square_ids(ids, center_id, rows, columns, square_side):
    new_ids = []
    max_dist = (square_side + 1) // 2
    center_y, center_x = get_position_from_index(center_id, columns)

    corner_x = center_x - max_dist + 1
    corner_y = center_y - max_dist + 1

    for i in range(len(ids)):
        el_y, el_x = get_position_from_index(ids[i], columns)

        # так как мир круглый, нужно смотреть расстояния в обе стороны осей
        dist_y = get_distance(el_y, center_y, rows)
        dist_x =  get_distance(el_x, center_x, columns)

        # пропускаем все элементы расположенные слишком далеко
        if dist_x >= max_dist or dist_y >= max_dist:
            continue
        # сдвигаем так, чтобы ось была в левом верхнем углу нового квадрата, где элемент с center_id в центре
        new_y, new_x = shifted_element_y_x(ids[i], corner_x, corner_y, rows, columns)
        new_ids.append(get_index_from_position(new_y, new_x, square_side))

    return new_ids


def transform_to_rows_x_rows(observation: Observation, prev_heads, rows, columns):
    observation = observation_copy(observation)
    index = observation.index
    head_id = observation.geese[index][0]

    if prev_heads != [-1, -1, -1, -1]:
        prev_heads = transform_to_central_square_ids(prev_heads, head_id, rows, columns, rows)
    # put the goose head into the center
    for i in range(len(observation.geese)):
        observation.geese[i] = transform_to_central_square_ids(observation.geese[i], head_id, rows, columns, rows)
    observation['food'] = transform_to_central_square_ids(observation.food, head_id, rows, columns, rows)

    return observation, prev_heads


# ROTATE
# change all ids as if we turned our picture 90 deg. anticlockwise k times
# it only works for squares...
def rotate_ids_90(ids, rows, columns, k):
    if k == 0:
        return ids.copy()
    res = []
    for j in range(k):
        res = []
        for i in range(len(ids)):
            y, x = get_position_from_index(ids[i], columns)
            new_y = rows - 1 - x
            new_x = y
            res.append(get_index_from_position(new_y, new_x, columns))
        ids = res

    return res


def observation_copy(observation: Observation):
    return Observation(
            {
                'index': observation.index,
                'geese': [x.copy() for x in observation.geese],
                'step': observation.step,
                'food': observation.food.copy(),
                'remainingOverageTime': observation.remaining_overage_time
            })


def set_head_to_the_north(observation: Observation, prev_heads, configuration: Configuration):
    observation = observation_copy(observation)
    prev_heads = prev_heads.copy()

    if observation.step == 0:
        return observation, prev_heads

    rows, columns = configuration.rows, configuration.columns
    index = observation.index

    if len(prev_heads) < index:
        return observation, prev_heads

    # get action that led to this observation
    # so it will show us the direction of the goose head
    # turn the picture so goose looks in the north direction
    rotates_num = get_north_head_rot_num(observation, prev_heads, rows, columns)
    # rotate geese's positions
    for i in range(len(observation.geese)):
        observation.geese[i] = rotate_ids_90(observation.geese[i].copy(), rows, columns, rotates_num)

    observation['food'] = rotate_ids_90(observation.food, rows, columns, rotates_num)
    prev_heads = rotate_ids_90(prev_heads, rows, columns, rotates_num)

    return observation, prev_heads


def get_north_head_rot_num(observation: Observation, prev_heads, rows, columns):
    index = observation.index
    head_y, head_x = get_position_from_index(observation.geese[index][0], columns)

    if len(prev_heads) == 0:
        return 0

    prev_head_y, prev_head_x = get_position_from_index(prev_heads[index], columns)
    rotates_num = 0
    if (head_x - prev_head_x + columns) % columns == 1:  # prev_action = 'EAST'
        rotates_num = 1
    elif (head_x - prev_head_x + columns) % columns == columns - 1:  # prev_action = 'WEST'
        rotates_num = 3
    elif (head_y - prev_head_y + rows) % rows == rows - 1:  # prev_action = 'NORTH'
        rotates_num = 0
    elif (head_y - prev_head_y + rows) % rows == 1:  # prev_action = 'SOUTH'
        rotates_num = 2

    return rotates_num


def get_position_from_index(index, columns):
    row = index // columns
    col = index % columns
    return row, col


def get_index_from_position(row, col, columns):
    return row * columns + col


def find_new_head_position(head_row, head_col, action, rows, columns):
    if action == 0:  # north
        new_row, new_col = (head_row + rows - 1) % rows, head_col
    elif action == 1:  # east
        new_row, new_col = head_row, (head_col + 1) % columns
    elif action == 2:  # south
        new_row, new_col = (head_row + 1) % rows, head_col
    else:  # west
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
    state[0, ids[0]] = 1  # goose head
    if len(ids) > 1:
        state[1, ids[1:-1]] = 1  # goose body
        state[2, ids[-1]] = 1  # goose tail
    if step != 0:
        state[3, prev_head] = 1  # goose head one step before
    return state


def get_features(observation: Observation, prev_heads, configuration: Configuration):
    rows, columns = configuration.rows, configuration.columns
    geese = observation.geese
    index = observation.index
    step = observation.step
    # convert indices to locations
    locations = np.zeros((len(geese), 4, rows * columns))
    for i, g in enumerate(geese):
        locations[i] = ids2locations(g, prev_heads[i], step, rows, columns)
    if index != 0:  # swap rows for player locations to be in first channel
        locations[[0, index]] = locations[[index, 0]]
    # put locations into features
    features = np.zeros((FEATURE_CHANNELS, rows * columns))
    for k in range(4):
        # склеить все части тела для каждого гуся (кроме предыдущей головы)
        features[k] = np.sum(locations[k][:3], 0)
    # склеить соответствующие состояния всех гусей (либо головы всех, либо тело, либо хвосты, либо пред. головы)
    for k in range(3):
        features[k + 4] = np.sum(locations[:, k], 0)

    # только пред. голова наешего героя
    features[-5, :] = locations[index][3]

    features[-4, observation.food] = 1  # food channel
    features[-3, :] = (step % configuration.hunger_rate) / configuration.hunger_rate  # hunger danger channel
    features[-2, :] = step / configuration.episode_steps  # timesteps channel
    features[-1, :] = float((step + 1) % configuration.hunger_rate == 0)  # hunger milestone indicator
    features = torch.Tensor(features).reshape(-1, rows, columns)
    # roll
    head_id = geese[index][0]
    head_row = head_id // columns
    head_col = head_id % columns
    features = torch.roll(features, ((rows // 2) - head_row, (columns // 2) - head_col), dims=(-2, -1))
    return features


def get_square_features_params(configuration: Configuration):
    # Channels, Height, Width
    return FEATURE_CHANNELS, configuration.rows, configuration.rows


def get_features_params(configuration: Configuration):
    # Channels, Height, Width
    return FEATURE_CHANNELS, configuration.rows, configuration.columns


def plot_features(features, title="Figure"):
    fig, axs = plt.subplots(3, 4, figsize=(20, 10))
    for i in range(3):
        for j in range(4):
            sns.heatmap(features[i * 4 + j], ax=axs[i, j], cmap='Blues',
                        vmin=0, vmax=1, linewidth=2, linecolor='black', cbar=False)
    plt.title(title)
    plt.show()


def get_example():
    observation = {'step': 104, 'index': 0, 'geese': [[46, 47, 36, 37, 48, 59, 58, 69],
                                                      [5, 71, 72, 6, 7, 73, 62, 61, 50, 51, 52, 63, 64, 53, 54],
                                                      [12, 11, 21, 20, 19, 8, 74, 75, 76, 65, 55, 56, 67, 1],
                                                      [23, 22, 32, 31, 30, 29, 28, 17, 16, 27, 26, 15, 14, 13, 24]],
                   'food': [45, 66], 'remainingOverageTime': 1}
    prev_heads = [47, 71, 11, 22]

    return Observation(observation), prev_heads


# TESTS

def test_transform_to_central_square_ids_1():
    ids = [46, 47, 36, 37, 48, 59, 58, 69]
    actual = transform_to_central_square_ids(ids, 46, 7, 11, 7)
    expected = [24, 25, 18, 19, 26, 33, 32, 39]
    return actual == expected


# some ids out of the area
def test_transform_to_central_square_ids_2():
    ids = [11, 6, 7, 40, 71]
    actual = transform_to_central_square_ids(ids, 46, 7, 11, 7)
    expected = [1, 41]
    return actual == expected


def test_rotate_90_1():
    ids = [24, 0, 6, 42, 48]
    expected = [24, 42, 0, 48, 6]
    actual = rotate_ids_90(ids, 7, 7, 1)
    return actual == expected


def test_rotate_90_2():
    ids = [9, 40, 2]
    expected = [29, 12, 28]
    actual = rotate_ids_90(ids, 7, 7, 1)
    return actual == expected


def test_set_head_to_north_1():
    observation = {'step': 104, 'index': 0, 'geese': [[24, 25, 32, 33],
                                                      [0, 1, 2, 3],
                                                      [21, 28, 35, 42],
                                                      [48]],
                   'food': [8, 33], 'remainingOverageTime': 1}
    prev_heads = [25, 1, 28, 42]
    config = Configuration({"rows": 7, "columns": 7, 'hunger_rate': 40, 'episodeSteps': 200})

    expected_obs = {'step': 104, 'index': 0, 'geese': [[24, 31, 30, 37],
                                                       [6, 13, 20, 27],
                                                       [3, 2, 1, 0],
                                                       [42]],
                    'food': [12, 37], 'remainingOverageTime': 1}

    expected_heads = [31, 13, 2, 0]
    actual_obs, actual_heads = set_head_to_the_north(Observation(observation), prev_heads, config)

    return actual_obs == expected_obs and actual_heads == expected_heads


# поворот не нужен, так как смотрит на север
def test_get_square_features_1():
    config = Configuration({"rows": 7, "columns": 11, 'hunger_rate': 40, 'episodeSteps': 200})
    prev_heads = [9, 23, 26, 20]
    obs = Observation({'remainingOverageTime': 60,
                       'step': 24,
                       'geese': [[75, 9, 8, 7], [24, 23, 34, 45], [], [21, 20]],
                       'food': [32, 64],
                       'index': 0})

    expected_features = np.zeros((FEATURE_CHANNELS, config.rows * config.rows))

    # bodies
    expected_features[0][[24, 31, 30, 29]] = 1
    expected_features[1][[48, 6, 13]] = 1
    expected_features[2][:] = 0
    expected_features[3][[39, 38]] = 1

    expected_features[4][[24, 48, 39]] = 1  # головы
    expected_features[5][[31, 30, 6]] = 1  # серединки
    expected_features[6][[29, 13, 38]] = 1  # хвосты
    expected_features[7][[31]] = 1  # предыдущая голова нашего героя

    expected_features[8][[46, 17]] = 1  # еда

    expected_features[9][:] = (obs.step % config.hunger_rate) / config.hunger_rate  # hunger danger channel
    expected_features[10, :] = obs.step / config.episode_steps  # timesteps channel
    expected_features[11, :] = float((obs.step + 1) % config.hunger_rate == 0)  # hunger milestone indicator
    expected_features = np.reshape(expected_features, (12, 7, 7))
    expected_features = torch.tensor(expected_features, dtype=torch.float)

    actual_features = get_square_features(obs, prev_heads, config)

    return torch.all(actual_features == expected_features)


# поворот нужен, так как смотрит на запад
def test_get_square_features_2():
    config = Configuration({"rows": 7, "columns": 11, 'hunger_rate': 40, 'episodeSteps': 200})
    prev_heads = [76, 23, 26, 20]
    obs = Observation({'remainingOverageTime': 60,
                       'step': 24,
                       'geese': [[75, 76, 10, 21], [24, 23, 34, 45], [], [21, 20]],
                       'food': [32, 64],
                       'index': 0})

    expected_features = np.zeros((FEATURE_CHANNELS, config.rows * config.rows))

    # bodies
    expected_features[0][[24, 31, 30, 29]] = 1
    expected_features[1][[42, 48, 47]] = 1
    expected_features[2][:] = 0
    expected_features[3][[29, 22]] = 1

    expected_features[4][[24, 42, 29]] = 1  # головы
    expected_features[5][[31, 30, 48]] = 1  # серединки
    expected_features[6][[29, 47, 22]] = 1  # хвосты
    expected_features[7][[31]] = 1  # предыдущая голова нашего героя

    expected_features[8][[28, 25]] = 1  # еда

    expected_features[9][:] = (obs.step % config.hunger_rate) / config.hunger_rate  # hunger danger channel
    expected_features[10, :] = obs.step / config.episode_steps  # timesteps channel
    expected_features[11, :] = float((obs.step + 1) % config.hunger_rate == 0)  # hunger milestone indicator
    expected_features = np.reshape(expected_features, (12, 7, 7))
    expected_features = torch.tensor(expected_features, dtype=torch.float)

    actual_features = get_square_features(obs, prev_heads, config)

    return torch.all(actual_features == expected_features)


# поворот нужен, так как смотрит на юг
# при этом левый угол вырезаемого экрана в левее нашей картинки (центр близко началу координат)
def test_get_square_features_3():
    config = Configuration({"rows": 7, "columns": 11, 'hunger_rate': 40, 'episodeSteps': 200})
    prev_heads = [0, 70, 26, 67]
    obs = Observation({'remainingOverageTime': 60,
                       'step': 24,
                       'geese': [[11, 0], [4, 70, 59], [25], [66, 67, 68]],
                       'food': [20, 63],
                       'index': 0})

    expected_features = np.zeros((FEATURE_CHANNELS, config.rows * config.rows))

    # bodies
    expected_features[0][[24, 31]] = 1
    expected_features[1][:] = 0
    expected_features[2][[14]] = 1
    expected_features[3][[38, 37, 36]] = 1

    expected_features[4][[24, 14, 38]] = 1  # головы
    expected_features[5][[37]] = 1  # серединки
    expected_features[6][[31, 36]] = 1  # хвосты
    expected_features[7][[31]] = 1  # предыдущая голова нашего героя

    expected_features[8][[26, 48]] = 1  # еда

    expected_features[9][:] = (obs.step % config.hunger_rate) / config.hunger_rate  # hunger danger channel
    expected_features[10, :] = obs.step / config.episode_steps  # timesteps channel
    expected_features[11, :] = float((obs.step + 1) % config.hunger_rate == 0)  # hunger milestone indicator
    expected_features = np.reshape(expected_features, (12, 7, 7))
    expected_features = torch.tensor(expected_features, dtype=torch.float)

    actual_features = get_square_features(obs, prev_heads, config)

    return torch.all(actual_features == expected_features)


def run_all_tests():
    print("Test 1: {0}".format(test_transform_to_central_square_ids_1()))
    print("Test 2: {0}".format(test_transform_to_central_square_ids_2()))
    print("Test 3: {0}".format(test_rotate_90_1()))
    print("Test 4: {0}".format(test_rotate_90_2()))
    print("Test 5: {0}".format(test_set_head_to_north_1()))
    print("Test 6: {0}".format(test_get_square_features_1()))
    print("Test 7: {0}".format(test_get_square_features_2()))
    print("Test 8: {0}".format(test_get_square_features_3()))


if __name__ == '__main__':
    run_all_tests()
    ...

    # obs, prev_heads = get_example()
    # env = make("hungry_geese", debug=False)
    # new_config = {"rows": 7, "columns": 7, 'hunger_rate': 40, 'episodeSteps': 200}
    # obs, prev_heads = transform_to_rows_x_rows(obs, prev_heads, 7, 11)
    #
    # init_features = get_features(obs, new_config, prev_heads)
    # obs, prev_heads = set_head_to_the_north(obs, prev_heads, new_config)
    # rot_features = get_features(obs, new_config, prev_heads)
    #
    # plot_features(init_features.detach().numpy(), title="Init")
    # plot_features(rot_features.detach().numpy(), title="Rotated")
