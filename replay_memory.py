from collections import namedtuple
import random
import numpy
import torch
import numpy as np
import math
from goose_tools import get_eps


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    MIN_POSSIBLE_PRIORITY = 1e-5
    write = 0
    min_priority = 0xffffffff

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        if p == 0:
            p = self.MIN_POSSIBLE_PRIORITY

        idx = self.write + self.capacity - 1
        self.pending_idx.add(idx)

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

        if self.min_priority > p:
            self.min_priority = p

    # update priority
    def update(self, idx, p):
        if idx not in self.pending_idx:
            return
        self.pending_idx.remove(idx)
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        self.pending_idx.add(idx)
        return (idx, self.data[dataIdx], self.tree[idx])


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        # self.memory = deque([], maxlen=capacity)
        # self.memory = queue.PriorityQueue(capacity)
        self.sum_tree = SumTree(capacity)
        self.capacity = capacity

    def push(self, priority, *args):
        """Save a transition"""
        self.sum_tree.add(priority, Transition(*args))

    def get(self, batch_size):
        if self.sum_tree.n_entries == 0:
            return None

        sample = []
        for i in range(batch_size):
            s = random.random() * self.sum_tree.total()
            sample.append(self.sum_tree.get(s))

        return sample

    def update(self, idx, p):
        self.sum_tree.update(idx, p)

    def __len__(self):
        return self.sum_tree.n_entries


def get_max_annealing_parameter(replay_memory, step, max_step):
    return annealing_parameter(replay_memory, replay_memory.sum_tree.min_priority,
                               step, max_step)


def annealing_parameter(replay_memory, priority, step, max_step):
    N = len(replay_memory)
    beta = get_eps(0.2, 1, step, max_step)
    prob = priority / replay_memory.sum_tree.total()

    omega = math.pow(1 / N / prob, beta)

    return omega


def update_priorities(idx_list, priorities, memory):
    for i in range(len(idx_list)):
        memory.update(idx_list[i], priorities[i])


def calculate_priority(reward, prev_state, state, action, target, policy, gamma):
    # R_j + gamma_j * Q_target(S_j, argmax_a(S_j, a)) - Q(S_{j - 1}, A_{j - 1})
    with torch.no_grad():
        q1 = policy(prev_state.unsqueeze(0)).squeeze(0)[action]
        q2 = reward + gamma * np.argmax(target(state.unsqueeze(0)).squeeze(0)) if state is not None else reward
        return np.abs(q2 - q1)


# В оригинальной статье Prioritized Experience Replay работают с одним приоритетом и для него одного вычисляется
# коэффициент отжига (annealing parameter), на который домнажается весь градиент
# Здесь же попытка сделать это для батчей, просто берется среднее всех коэффициентов
# так же подсчет максимального коэффициента отжига считается на основании минимального
# встретившегося в памяти приоритета не на данный момент, а в течение всей работы с этой коллекцией
def get_priority_weight(priorities, replay_memory, step, max_step):
    max_annealing_param = get_max_annealing_parameter(replay_memory, step, max_step)
    map_func = lambda x: \
        math.pow(max_annealing_param, -1) * annealing_parameter(replay_memory,
                                                                x,
                                                                step,
                                                                max_step)
    weights_c = list(map(map_func, priorities))
    return np.average(weights_c)
