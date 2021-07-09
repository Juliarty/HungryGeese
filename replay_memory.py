import random
import numpy as np
import math
from sum_tree import SumTree


class ReplayMemory(object):
    MAXIMAL_PRIORITY = 100

    def __init__(self, capacity):
        self.sum_tree = SumTree(capacity)
        self.capacity = capacity
        self.n_data_with_max_priority = 0

    def push(self, priority, data):
        """Save a transition"""
        self.sum_tree.add(priority, data)
        if priority == ReplayMemory.MAXIMAL_PRIORITY:
            self.n_data_with_max_priority += 1

    def get(self, batch_size):
        if self.sum_tree.n_entries == 0:
            return None

        sample = []
        for i in range(batch_size):
            s = random.random() * self.sum_tree.total()
            sample.append(self.sum_tree.get(s))

        return sample

    def __len__(self):
        return self.sum_tree.n_entries

    def update_priorities(self, idx_list, priorities):
        for i in range(len(idx_list)):
            if self.sum_tree.tree[idx_list[i]] == ReplayMemory.MAXIMAL_PRIORITY:
                self.n_data_with_max_priority = np.max(self.n_data_with_max_priority - 1, 0)
            self._update(idx_list[i], priorities[i])

    def is_there_new_experience(self):
        return self.n_data_with_max_priority > 0

    def _update(self, idx, p):
        self.sum_tree.update(idx, p)


# В оригинальной статье Prioritized Experience Replay работают с одним приоритетом и для него одного вычисляется
# коэффициент отжига (annealing parameter), на который домнажается весь градиент
# Здесь же попытка сделать это для батчей, просто берется среднее всех коэффициентов
# так же подсчет максимального коэффициента отжига считается на основании минимального
# встретившегося в памяти приоритета не на данный момент, а в течение всей работы с этой коллекцией
def get_priority_weight(priorities, replay_memory, beta):
    max_annealing_param = get_max_annealing_parameter(replay_memory, beta)
    map_func = lambda x: \
        math.pow(max_annealing_param, -1) * annealing_parameter(replay_memory,
                                                                x,
                                                                beta)
    weights_c = list(map(map_func, priorities))
    return np.average(weights_c)


def get_max_annealing_parameter(replay_memory, beta):
    return annealing_parameter(replay_memory, replay_memory.sum_tree.min_priority, beta)


def annealing_parameter(replay_memory, priority, beta):
    N = len(replay_memory)
    prob = priority / replay_memory.sum_tree.total()
    omega = math.pow(1 / N / prob, beta)
    return omega
