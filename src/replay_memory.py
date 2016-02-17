import numpy as np
import random


# Set for performing fast addition, deletion and taking random sample of elements.
# Can store integer values in range [0, max_size). add/delete operations take O(1) time,
# while get_random_elements takes as much as random.sample
class FastSet(object):
    def __init__(self, max_size):
        self.where = np.empty([max_size], dtype=np.int32)
        self.where.fill(-1)
        self.container = np.empty([max_size], dtype=np.int32)
        self.size = 0
        self.max_size = max_size

    def add(self, el):
        assert 0 <= el < self.max_size
        if self.where[el] != -1:
            return
        self.where[el] = self.size
        self.container[self.size] = el
        self.size += 1

    def delete(self, el):
        assert 0 <= el < self.max_size
        if self.where[el] == -1:
            return
        last_el = self.container[self.size - 1]
        new_pos = self.where[el]
        self.container[new_pos] = last_el
        self.where[last_el] = new_pos
        self.where[el] = -1
        self.size -= 1

    def find(self, el):
        assert 0 <= el < self.max_size
        return self.where[el] != -1

    def get_random_elements(self, count):
        return random.sample(self.container[: self.size], count)

    def get_content(self):
        return self.container[:self.size]


class ReplayMemoryManager(object):
    def __init__(self, image_height, image_width, state_frames, max_size, reward_clip_min, reward_clip_max):
        self.max_size = max_size
        self.state_frames = state_frames
        self.screen_queue = np.empty([max_size, image_height, image_width], dtype=np.uint8)
        self.reward_queue = np.empty([max_size], dtype=np.float32)
        self.action_queue = np.empty([max_size], dtype=np.int8)
        self.termination_queue = np.empty([max_size], dtype=np.bool)
        self.legit_state = FastSet(max_size)
        self.size = 0
        self.end = 0
        self.frames_from_last_termination = 0
        self.image_height = image_height
        self.image_width = image_width
        self.state_frames = state_frames
        self.min_reward = reward_clip_min
        self.max_reward = reward_clip_max

    def state_at(self, index, next_state=False):
        if not next_state:
            assert self.legit_state.find(index)
        else:
            assert self.legit_state.find((index - 1 + self.size) % self.size)
        assert self.size >= self.state_frames
        if index >= self.state_frames:
            res = self.screen_queue[index - self.state_frames: index]
        else:
            res = np.concatenate([self.screen_queue[self.size - self.state_frames + index: self.size],
                                  self.screen_queue[0: index]])
        return res

    def last_state(self):
        assert self.size > 0
        return self.state_at((self.end - 1 + self.size) % self.size)

    def add_experience(self, prev_screen, action, reward, game_over):
        reward = max(reward, min(reward, self.max_reward), self.min_reward)
        self.screen_queue[self.end] = prev_screen
        self.action_queue[self.end] = action
        self.reward_queue[self.end] = reward
        self.termination_queue[self.end] = game_over
        self.frames_from_last_termination += 1
        if self.frames_from_last_termination >= self.state_frames:
            self.legit_state.add(self.end)
        else:
            #todo: think about it
            #self.legit_state.delete(self.end)
            self.legit_state.add(self.end)
        if game_over:
            self.frames_from_last_termination = 0

        if self.size == self.max_size:
            if self.state_frames > 1:
                self.legit_state.delete((self.end + self.state_frames - 1) % self.max_size)
        else:
            self.size += 1
        self.end = (self.end + 1) % self.max_size

    def __len__(self):
        return self.size

    def get_batch(self, batch_size):
        indices = self.legit_state.get_random_elements(batch_size)
        return {'states_1': np.asarray([self.state_at(index) for index in indices]) / 128.0,
                'states_2': np.asarray([self.state_at((index + 1) % self.size, True) for index in indices]) / 128.0,
                'actions': self.action_queue[indices],
                'rewards': self.reward_queue[indices],
                'terminations': self.termination_queue[indices],
                'batch_size': batch_size}

    def save_to_file(self, filename):
        pass

    def load_from_file(self, filename):
        pass
