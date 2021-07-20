import random

from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, Observation, translate
from kaggle_environments.helpers import histogram

from actions_generator_strategy import AbstractActionsGeneratorStrategy
from goose_tools import get_enemy_head_adjacent_positions

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

    def generate_children(self, action_generator_strategy: AbstractActionsGeneratorStrategy, get_reward,
                          get_risk_penalty=None):
        result = []
        agent_actions_list = action_generator_strategy.get_actions(self.observation, self.prev_obs, self.prev_actions)
        for agent_actions in agent_actions_list:
            next_obs = get_next_observation(self.observation, agent_actions, self.prev_actions)
            instant_reward = get_reward(next_obs, self.observation)

            if get_risk_penalty is not None:
                instant_reward += get_risk_penalty(next_obs, self.observation)

            if action_generator_strategy.filter_observation(next_obs, self.observation):
                child = OracleTreeNode(self, next_obs, self.observation, agent_actions,
                                       self.accumulated_reward + instant_reward)
                result.append(child)
        return result

    def __str__(self):
        d = {
            "observation": self.observation,
            "prev_obs": self.prev_obs,
            "prev_actions": self.prev_actions
        }
        return str(d)


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
            return self._get_not_bad_action(self.root.observation,
                                            self.root.prev_obs,
                                            self.root.prev_actions,
                                            get_state,
                                            q_estimator)
        best_node = None
        max_prediction = -100000
        for node in self.depth_to_peers[self.depth]:
            state = get_state(node.observation, node.parent.observation).unsqueeze(0)
            prediction = float(node.accumulated_reward + q_estimator(state).squeeze().max())
            if prediction > max_prediction:
                best_node = node
                max_prediction = prediction
        # print(self._take_second_parent(best_node))
        return self._take_second_parent(best_node).prev_actions[best_node.observation['index']]

    def _get_not_bad_action(self, observation: Observation, prev_obs: Observation, prev_actions, get_state,
                            q_estimator):
        """
        Choose save action (he neither commits suicide nor does kamikadze moves), but probably moves to positions
        that are adjacent to other goose heads.
        :param state:
        :return:
        """
        # Don't move into any bodies
        bodies = {position for goose in observation.geese for position in goose[:-1]}

        position = observation.geese[observation.index][0]
        bad_action_idx_list = [
            action.value - 1 for action in Action
            for new_position in [translate(position, action, configuration.columns, configuration.rows)]
            if (new_position in bodies or (
                        prev_actions is not None and action == prev_actions[observation.index].opposite()))]

        head_adjacent_positions = get_enemy_head_adjacent_positions(observation)
        risky_action_idx_list = [
            action.value - 1 for action in Action
            for new_position in [translate(position, action, configuration.columns, configuration.rows)]
            if new_position in head_adjacent_positions]
        state = get_state(observation, prev_obs)

        q_values = q_estimator(state.unsqueeze(0)).squeeze()
        q_values[bad_action_idx_list] = -10000
        q_values[risky_action_idx_list] = -5000
        return Action(int(q_values.max(0)[1]) + 1)

    def _take_second_parent(self, node: OracleTreeNode):
        while node.parent.parent is not None:
            node = node.parent
        return node


# def get_next_observation(observation, agent_actions, prev_actions):
#     observation = Observation(observation)
#     next = Observation({'index': observation.index,
#                         'step': observation.step + 1,
#                         'remainingOverageTime': observation.remaining_overage_time,
#                         'food': observation.food.copy(),
#                         'geese': [[], [], [], []]})
#
#     starved = 0
#     if next.step > 0 and next.step % 40 == 0:
#         starved = 1
#
#     # move geese (they ate and starve)
#     for i in range(len(observation.geese)):
#         # already dead or took an opposite action
#         if len(observation.geese[i]) == 0 or \
#                 (prev_actions is not None and prev_actions[i] == agent_actions[i].opposite() and len(observation.geese[i]) )> 1:  # suicide
#             next.geese[i] = []
#             continue
#
#         next_head = translate(observation.geese[i][0], agent_actions[i], configuration.columns, configuration.rows)
#         # remove eaten food
#         ate = 1 if next_head in next.food else 0
#         if ate == 1:
#             next.food.remove(next_head)
#         # remove starved, do it in the end because they could hit someone before dead
#
#         next.geese[i] = ([next_head] + observation.geese[i][0:-1])
#
#         if ate:
#             next.geese[i] += [observation.geese[i][-1]]
#         if starved:
#             next.geese[i] = next.geese[i][:-1]
#
#     # remove dead bodies
#     # remove collided
#     for i in range(len(next.geese)):
#         if len(next.geese[i]) == 0:
#             continue
#         collided = False
#         for j in range(i + 1, len(next.geese)):
#             if len(next.geese[j]) == 0:
#                 continue
#             if next.geese[i][0] == next.geese[j][0]:
#                 next.geese[j] = []
#                 collided = True
#         if collided:
#             next.geese[i] = []
#     # remove body hit:
#     for i in range(len(next.geese)):
#         if len(next.geese[i]) == 0:
#             continue
#         # we also consider self collision
#         for j in range(len(next.geese)):
#             if len(next.geese[j]) == 0:
#                 continue
#             if next.geese[i][0] in next.geese[j][1:]:  # we already checked head collisions
#                 next.geese[i] = []
#                 break
#
#     if len(next.food) == 0:
#         next.food.append(random.choice(range(76)))
#
#     return next


def get_risk_penalty(observation, prev_obs):
    if len(observation.geese[observation.index]) == 0:
        return 0    # there is no penalty for dead...

    # collisions
    head_adjacent_positions = get_enemy_head_adjacent_positions(prev_obs)
    if observation.geese[observation.index][0] in head_adjacent_positions:
        return -1

    # attempts to catch the tail of an eating goose
    for i in range(len(observation.geese)):
        if i == observation.index or len(observation.geese[i]) == 0:
            continue
        if observation.geese[i] not in observation.food:
            continue
        head = observation.geese[observation.index][0]
        prev_enemy_tail = prev_obs.geese[i][-1]
        if prev_enemy_tail == head:
            print("penalty")
            return -1

    return 0


def get_next_observation(observation, actions, prev_actions):
    rows = configuration.rows
    columns = configuration.columns
    next = Observation(observation.copy())
    next['step'] += 1
    geese = [goose[:] for goose in next.geese]
    food = next.food.copy()
    next['geese'] = geese
    next['food'] = food
    # If there is no last state, reuse current state so that current action is never the opposite of the last action.
    # Apply the actions from active agents.
    for index in range(len(next.geese)):
        # if agent.status != "ACTIVE":
        #     if agent.status != "INACTIVE" and agent.status != "DONE":
        #         # ERROR, INVALID, or TIMEOUT, remove the goose.
        #         geese[index] = []
        #     continue
        if len(geese[index]) == 0:
            continue

        action = actions[index]

        # Check action direction
        if prev_actions is not None and prev_actions[index] == action.opposite():
            # agent.status = "DONE"
            geese[index] = []
            continue

        goose = geese[index]
        head = translate(goose[0], action, columns, rows)

        # Consume food or drop a tail piece.
        if head in food:
            food.remove(head)
        else:
            goose.pop()

        # Self collision.
        if head in goose:
            # env.debug_print(f"Body Hit: {agent.observation.index, action, head, goose}")
            # agent.status = "DONE"
            geese[index] = []
            continue

        # while len(goose) >= configuration.max_length:
        #     # Free a spot for the new head if needed
        #     goose.pop()
        # Add New Head to the Goose.
        goose.insert(0, head)

        # If hunger strikes remove from the tail.
        if next.step % configuration.hunger_rate == 0:
            if len(goose) > 0:
                goose.pop()
            if len(goose) == 0:
                # env.debug_print(f"Goose Starved: {action}")
                # agent.status = "DONE"
                continue

    goose_positions = histogram(
        position
        for goose in geese
        for position in goose
    )

    # Check for collisions.
    for index in range(len(next.geese)):
        goose = geese[index]
        if len(goose) > 0:
            head = geese[index][0]
            if goose_positions[head] > 1:
                # env.debug_print(f"Goose Collision: {agent.action}")
                # agent.status = "DONE"
                geese[index] = []

    # Add food if min_food threshold reached.
    needed_food = 2 - len(food)
    if needed_food > 0:
        collisions = {
            position
            for goose in geese
            for position in goose
        }
        available_positions = set(range(rows * columns)).difference(collisions).difference(food)
        # Ensure we don't sample more food than available positions.
        needed_food = min(needed_food, len(available_positions))
        food.extend(random.sample(available_positions, needed_food))

    return next