#   Строим картину будущего мира, проигрывая несколько ходов вперёд
#   1. Проигрывать наш лучший ход, согласно предсказанию оракул
#       a. Выбрать лучший ход из позиции
#       b. Выбрать все движения за врагов, обрезать хвосты
#       c. Добавить ход к дереву
#       d. Повторить n_steps - 1 h раз

from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, Observation, translate, \
    adjacent_positions

from actions_generator_strategy import AbstractActionsGeneratorStrategy

configuration = Configuration({"rows": 7, "columns": 11, 'hunger_rate': 40, 'episodeSteps': 200})


class Oracle:
    def __init__(self, q_estimator, actions_generator_strategy, get_state, get_reward):
        self.action_generator_strategy = actions_generator_strategy
        self.q_estimator = q_estimator
        self.get_state = get_state
        self.get_reward = get_reward

    def tell_best_action(self, observation, prev_obs, prev_actions, depth):
        root = OracleTreeNode(None, observation=observation, prev_obs=prev_obs, prev_actions=prev_actions)
        ot = OracleTree(root)
        # careful look
        ot.look_into_the_future(self.action_generator_strategy, self.get_reward, get_risk_penalty)

        for _ in range(depth - 1):
            ot.look_into_the_future(self.action_generator_strategy, self.get_reward)

        return ot.get_best_action(self.get_state, self.q_estimator)


class OracleTreeNode:
    def __init__(self, parent, observation, prev_obs, prev_actions, accumulated_reward=0):
        self.accumulated_reward = accumulated_reward
        self.parent = parent
        self.observation = observation
        self.prev_obs = prev_obs
        self.prev_actions = prev_actions

    def generate_children(self, action_generator_strategy: AbstractActionsGeneratorStrategy, get_reward, get_risk_penalty=None):
        result = []
        agent_actions_list = action_generator_strategy.get_actions(self.observation, self.prev_obs, self.prev_actions)
        for agent_actions in agent_actions_list:
            next_obs = get_next_observation(self.observation, agent_actions, self.prev_actions)
            instant_reward = get_reward(next_obs, self.observation)

            if get_risk_penalty is not None:
                instant_reward += get_risk_penalty(next_obs, self.observation)

            if is_observation_good(next_obs):
                child = OracleTreeNode(self, next_obs, self.observation, agent_actions, self.accumulated_reward + instant_reward)
                result.append(child)
        return result


class OracleTree:
    def __init__(self, root):
        self.root = root
        self.depth_to_peers = {0: [self.root]}
        self.depth = 0

    def look_into_the_future(self, action_generator_strategy: AbstractActionsGeneratorStrategy, get_reward,
                             get_penalty=None):
        next_peers = []
        for node in self.depth_to_peers[self.depth]:
            next_peers += node.generate_children(action_generator_strategy, get_reward, get_penalty)
        self.depth_to_peers[self.depth + 1] = next_peers
        self.depth += 1

    def get_best_action(self, get_state, q_estimator):
        if len(self.depth_to_peers[self.depth]) == 0:
            # take our best action, oracle can't help here
            state = get_state(self.root.observation, self.root.prev_obs).unsqueeze(0)
            action_index = int(q_estimator(state).max(1)[1].view(1)[0])
            return Action(action_index + 1)
        best_node = None
        max_prediction = -100000
        for node in self.depth_to_peers[self.depth]:
            state = get_state(node.observation, node.parent.observation).unsqueeze(0)
            prediction = float(node.accumulated_reward + q_estimator(state).max(1)[0].view(1)[0])
            if prediction > max_prediction:
                best_node = node
                max_prediction = prediction

        return self._take_second_parent(best_node).prev_actions[best_node.observation['index']]

    def _take_second_parent(self, node: OracleTreeNode):
        while node.parent.parent is not None:
            node = node.parent
        return node


def get_next_observation(observation, agent_actions, prev_actions):
    observation = Observation(observation)
    next = Observation({'index': observation.index,
                        'step': observation.step + 1,
                        'remainingOverageTime': observation.remaining_overage_time,
                        'food': observation.food.copy(),
                        'geese': []})

    starved = 0
    if next.step > 0 and next.step % 40 == 0:
        starved = 1
    # move geese (they ate and starve)
    for i in range(len(observation.geese)):
        # already dead or took an opposite action
        if len(observation.geese[i]) == 0 \
                or \
                (prev_actions is not None and Action(prev_actions[i]).value == Action(agent_actions[i]).opposite().value):  # suicide
            next.geese.append([])
            continue

        next_head = translate(observation.geese[i][0], agent_actions[i], configuration.columns, configuration.rows)
        # remove eaten food
        ate = 1 if next_head in next.food else 0
        if ate == 1:
            next.food.remove(next_head)
        # remove starved, do it in the end because they could hit someone before dead

        next.geese.append([next_head] + observation.geese[i][0:-1])

        if ate:
            next.geese[i] += [observation.geese[i][-1]]
        if starved:
            next.geese[i] = next.geese[i][:-1]

    # remove dead bodies
    # remove collided
    for i in range(len(next.geese)):
        if len(next.geese[i]) == 0:
            continue
        collided = False
        for j in range(i + 1, len(next.geese)):
            if len(next.geese[j]) == 0:
                continue
            if next.geese[i][0] == next.geese[j][0]:
                next.geese[j] = []
                collided = True
        if collided:
            next.geese[i] = []
    # remove body hit:
    for i in range(len(next.geese)):
        if len(next.geese[i]) == 0:
            continue
        # we also consider self collision
        for j in range(len(next.geese)):
            if len(next.geese[j]) == 0:
                continue
            if next.geese[i][0] in next.geese[j][1:]:   # we already checked head collisions
                next.geese[i] = []
                break

    return next


# at least we survive, may be sth else
def is_observation_good(observation: Observation):
    result = True
    if len(observation.geese[observation.index]) == 0:
        result = False
    return result


def get_risk_penalty(observation, prev_obs):
    if len(observation.geese[observation.index]) == 0:
        return 0

    opponents = [
        goose
        for index, goose in enumerate(prev_obs.geese)
        if index != prev_obs.index and len(goose) > 0
    ]

    head_adjacent_positions = {
        opponent_head_adjacent
        for opponent in opponents if len(opponent) > 0
        for opponent_head in [opponent[0]]
        for opponent_head_adjacent in adjacent_positions(opponent_head, configuration.columns, configuration.rows)
    }

    if observation.geese[observation.index][0] in head_adjacent_positions:
        return -1

    return 0