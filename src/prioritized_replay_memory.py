import numpy as np
import random

EPSILON = 0.000001


class StatePriority(object):
    def __init__(self, max_size):
        self.base_size = 1 << int(np.ceil(np.log2(max_size)))
        self.sum_tree = np.empty([self.base_size * 2])
        self.max_tree = np.empty([self.base_size * 2])

    # used for inserting and updating
    def insert(self, el, prob, add_epsilon=True):
        start = self.base_size + el
        self.sum_tree[start] = prob + EPSILON if add_epsilon else 0
        self.max_tree[start] = self.sum_tree[start]
        start /= 2
        while start >= 1:
            self.sum_tree[start] = self.sum_tree[start * 2] + self.sum_tree[start * 2 + 1]
            self.max_tree[start] = max(self.max_tree[start * 2], self.max_tree[start * 2 + 1])

    def get_random(self):
        x = random.random() * self.sum_tree[1]
        start = 1
        while start < self.base_size:
            if self.sum_tree[start * 2] >= x:
                start *= 2
            else:
                start = start * 2 + 1
                x -= self.sum_tree[start - 1]
        return x - self.base_size

    def get_random_elements(self, count):
        return [self.get_random() for _ in range(count)]

    def delete(self, el):
        self.insert(el, 0, False)

    def find(self, el):
        return self.sum_tree[self.base_size + el] == 0

    def max(self):
        return self.max_tree[1]


class PrioritizedReplayMemoryManager(object):
    def __init__(self, image_height, image_width, max_size=150000, state_frames=4):
        self.max_size = max_size
        self.state_frames = state_frames
        self.screen_queue = np.empty([max_size, image_height, image_width], dtype=np.uint8)
        self.reward_queue = np.empty([max_size], dtype=np.float32)
        self.action_queue = np.empty([max_size], dtype=np.int8)
        self.termination_queue = np.empty([max_size], dtype=np.bool)
        self.size = 0
        self.end = 0
        self.frames_from_last_termination = 0
        self.image_height = image_height
        self.image_width = image_width
        self.state_frames = state_frames
        self.state_priority = StatePriority(max_size=max_size)

    def state_at(self, index, next_state=False):
        if not next_state:
            assert self.state_priority.find(index)
        else:
            assert self.state_priority.find((index - 1 + self.size) % self.size)
        assert self.size >= self.state_frames
        if index - self.state_frames >= 0:
            res = self.screen_queue[index - self.state_frames + 1: index + 1]
        else:
            res = np.concatenate([self.screen_queue[self.end - self.state_frames + index + 1: self.end],
                                  self.screen_queue[0: index + 1]])
        return res.transpose([1, 2, 0])

    def last_state(self):
        assert self.size > 0
        return self.state_at(self.end - 1)

    def add_experience(self, prev_screen, action, reward, game_over):
        self.screen_queue[self.end] = prev_screen
        self.action_queue[self.end] = action
        self.reward_queue[self.end] = reward
        self.termination_queue[self.end] = game_over
        self.frames_from_last_termination += 1
        if self.frames_from_last_termination >= self.state_frames:
            self.state_priority.add(self.end)
        else:
            #todo: think about it
#            self.legit_state.delete(self.end)
            self.state_priority.add(self.end)
        if game_over:
            self.frames_from_last_termination = 0

        if self.size == self.max_size:
            if self.state_frames > 1:
                self.state_priority.delete(self.end + self.state_frames - 1)
        else:
            self.size += 1
        self.end = (self.end + 1) % self.max_size

    def __len__(self):
        return self.size

    def get_batch(self, batch_size):
        indices = self.state_priority.get_random_elements(batch_size)
        return {'states_1': np.asarray([self.state_at(index) for index in indices]) / 128.0,
                'states_2': np.asarray([self.state_at((index + 1) % self.size, True) for index in indices]) / 128.0,
                'actions': self.action_queue[indices],
                'rewards': self.reward_queue[indices],
                'terminations': self.termination_queue[indices],
                'indices': indices,
                'batch_size': batch_size}
