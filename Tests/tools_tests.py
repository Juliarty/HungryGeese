from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Action

from goose_tools import get_prev_actions


def test_get_prev_actions_1():
    observation = Observation({'remainingOverageTime': 56.499099, 'index': 3, 'step': 60,
                               'geese': [[1], [4], [49], [75]], 'food': [53, 46]})
    prev_obs = Observation({'remainingOverageTime': 56.499099, 'index': 3, 'step': 60,
                               'geese': [[12], [3], [38], [76]], 'food': [53, 46]})
    expected = [Action(i) for i in range(1, 5)]
    actual = get_prev_actions(observation, prev_obs)
    return actual == expected


def test_get_prev_actions_2():
    observation = Observation({'remainingOverageTime': 56.499099, 'index': 3, 'step': 60,
                               'geese': [[1, 12, 23], [], [49], [75]], 'food': [53, 46]})
    prev_obs = Observation({'remainingOverageTime': 56.499099, 'index': 3, 'step': 60,
                               'geese': [[12, 23, 34], [], [38], [76]], 'food': [53, 46]})
    expected = [Action(1), Action(1), Action(3), Action(4)]
    actual = get_prev_actions(observation, prev_obs)
    return actual == expected


def run_tests():
    if not test_get_prev_actions_1():
        print("Error: Test 1 Get prev actions")

    if not test_get_prev_actions_2():
        print("Error: Test 1 Get prev actions")


if __name__ == "__main__":
    run_tests()
