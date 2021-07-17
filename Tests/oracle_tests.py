import torch
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, Observation, translate

from actions_generator_strategy import DeepQGeneratorStrategy
from feature_transform import AnnFeatureTransform
from oracle import get_next_observation, Oracle
from qestimator import GooseNetGoogle


def test_get_next_observation_1():
    result = True
    observation = Observation({'index': 2,
                               'step': 2,
                               'remainingOverageTime': 0,
                               'food': [3, 4],
                               'geese': [[48, 47], [33, 22], [27, 26], [3]]})
    prev_actions = [Action(2), Action(3), Action(2), Action(1)]
    agent_actions = [Action(2), Action(3), Action(2), Action(1)]
    actual_observation = get_next_observation(observation, agent_actions, prev_actions)
    expected_observation = Observation({'index': 2,
                                        'step': 3,
                                        'remainingOverageTime': 0,
                                        'food': [3, 4],
                                        'geese': [[49, 48], [44, 33], [28, 27], [69]]})

    return expected_observation == actual_observation


def test_get_next_observation_2_food():
    result = True
    observation = Observation({'index': 2,
                               'step': 2,
                               'remainingOverageTime': 0,
                               'food': [3, 49],
                               'geese': [[48, 47], [33, 22], [27, 26], [3]]})
    prev_actions = [Action(2), Action(3), Action(2), Action(1)]
    agent_actions = [Action(2), Action(3), Action(2), Action(1)]
    actual_observation = get_next_observation(observation, agent_actions, prev_actions)
    expected_observation = Observation({'index': 2,
                                        'step': 3,
                                        'remainingOverageTime': 0,
                                        'food': [3],
                                        'geese': [[49, 48, 47], [44, 33], [28, 27], [69]]})

    return expected_observation == actual_observation


def test_get_next_observation_3_collision():
    observation = Observation({'index': 2,
                               'step': 2,
                               'remainingOverageTime': 0,
                               'food': [3, 49],
                               'geese': [[48, 47], [50, 51], [27, 26], [3]]})
    prev_actions = [Action(2), Action(4), Action(2), Action(1)]
    agent_actions = [Action(2), Action(4), Action(2), Action(1)]
    actual_observation = get_next_observation(observation, agent_actions, prev_actions)
    expected_observation = Observation({'index': 2,
                                        'step': 3,
                                        'remainingOverageTime': 0,
                                        'food': [3],
                                        'geese': [[], [], [28, 27], [69]]})

    return expected_observation == actual_observation


def test_get_next_observation_4_body_hit():
    observation = Observation({'index': 2,
                               'step': 2,
                               'remainingOverageTime': 0,
                               'food': [3],
                               'geese': [[49, 48, 47], [61, 50, 51], [27, 26], [3]]})
    prev_actions = [Action(2), Action(3), Action(2), Action(1)]
    agent_actions = [Action(2), Action(3), Action(2), Action(1)]
    actual_observation = get_next_observation(observation, agent_actions, prev_actions)
    expected_observation = Observation({'index': 2,
                                        'step': 3,
                                        'remainingOverageTime': 0,
                                        'food': [3],
                                        'geese': [[], [72, 61, 50], [28, 27], [69]]})

    return expected_observation == actual_observation


def test_get_next_observation_5_body_hit():
    observation = Observation({'index': 2,
                               'step': 2,
                               'remainingOverageTime': 0,
                               'food': [3],
                               'geese': [[59, 60, 49, 48, 47], [61, 50, 51], [27, 26], [3]]})
    prev_actions = [Action(4), Action(3), Action(2), Action(1)]
    agent_actions = [Action(1), Action(3), Action(2), Action(1)]
    actual_observation = get_next_observation(observation, agent_actions, prev_actions)
    expected_observation = Observation({'index': 2,
                                        'step': 3,
                                        'remainingOverageTime': 0,
                                        'food': [3],
                                        'geese': [[], [72, 61, 50], [28, 27], [69]]})

    return expected_observation == actual_observation


def test_get_next_observation_6_opposite_action():
    observation = Observation({'index': 2,
                               'step': 2,
                               'remainingOverageTime': 0,
                               'food': [3],
                               'geese': [[59, 60, 49, 48, 47], [61, 50, 51], [27, 26], [3]]})
    prev_actions = [Action(4), Action(3), Action(2), Action(1)]
    agent_actions = [Action(2), Action(1), Action(4), Action(3)]
    actual_observation = get_next_observation(observation, agent_actions, prev_actions)
    expected_observation = Observation({'index': 2,
                                        'step': 3,
                                        'remainingOverageTime': 0,
                                        'food': [3],
                                        'geese': [[], [], [], []]})

    return expected_observation == actual_observation


# the bad situation, guy tries to survive by moving to the enemy tail that will eat food next step...
def test_oracle_tell_best_action():
    observation = Observation({'index': 1,
                               'step': 103,
                               'remainingOverageTime': 0,
                               'food': [8, 29],
                               'geese': [
                                   [30, 31, 42, 43, 54, 65, 55, 56, 57, 58, 69, 70, 71, 5, 4, 3, 2, 13, 12, 11, 21],
                                   [32, 22, 33, 44, 45, 34, 23, 24, 25, 26, 27, 38, 49, 50, 61], [], []]})
    prev_obs = Observation({'index': 1,
                            'step': 102,
                            'remainingOverageTime': 0,
                            'food': [8, 29],
                            'geese': [[31, 42, 43, 54, 65, 55, 56, 57, 58, 69, 70, 71, 5, 4, 3, 2, 13, 12, 11, 21, 32],
                                      [32, 22, 33, 44, 45, 34, 23, 24, 25, 26, 27, 38, 49, 50, 61, 62], [], []]})
    prev_actions = [Action(4), Action(4), Action(1), Action(1)]
    depth = 2

    net_path = "../Champions/75000_GooseNetGoogle_c14_h7_w11_DQN.net"
    net = GooseNetGoogle()
    net.load_state_dict(torch.load(net_path))
    oracle = Oracle(net, DeepQGeneratorStrategy(net, AnnFeatureTransform.get_state), AnnFeatureTransform.get_state,
                    AnnFeatureTransform.get_reward)
    actual_action = oracle.tell_best_action(observation, prev_obs, prev_actions, depth)
    expected_action = Action(1)

    return actual_action == expected_action


def test_oracle_tell_best_action_2():
    observation = Observation({
        "remainingOverageTime": 60,
        "step": 1,
        "geese": [[31, 32], [69], [9], [36]],
        "food": [11, 26],
        "index": 0
    })

    prev_obs = Observation({
        "remainingOverageTime": 60,
        "step": 0,
        "geese": [[32], [70], [75], [47]],
        "food": [11, 31],
        "index": 0
    })
    prev_actions = [Action(4), Action(4), Action(3), Action(1)]
    depth = 4

    net_path = "../Champions/75000_GooseNetGoogle_c14_h7_w11_DQN.net"
    net = GooseNetGoogle()
    net.load_state_dict(torch.load(net_path))
    oracle = Oracle(net, DeepQGeneratorStrategy(net, AnnFeatureTransform.get_state), AnnFeatureTransform.get_state,
                    AnnFeatureTransform.get_reward)
    actual_action = oracle.tell_best_action(observation, prev_obs, prev_actions, depth)
    expected_action = Action(1)

    return actual_action == expected_action


def run_tests():
    if not test_get_next_observation_1():
        print("Error: Test 1 Get next observation")
    if not test_get_next_observation_2_food():
        print("Error: Test 2 Get next observation with feeding")
    if not test_get_next_observation_3_collision():
        print("Error: Test 3 Get next observation with collision")
    if not test_get_next_observation_4_body_hit():
        print("Error: Test 4 Get next observation with enemy body hit")
    if not test_get_next_observation_5_body_hit():
        print("Error: Test 5 Get next observation with itself")
    if not test_get_next_observation_6_opposite_action():
        print("Error: Test 6 taking opposite action")
    if not test_oracle_tell_best_action():
        print("Error: Test 7 Oracle tell")
    if not test_oracle_tell_best_action_2():
        print("Error: Test 8 Oracle tell 2")

if __name__ == "__main__":
    run_tests()
