import random
import sys


class Agent(object):
    def __init__(self, game_handler, qnetwork, replay_memory, saver, monitor, train_freq, test_mode, batch_size=32,
                 save_every_x_episodes=100):
        self.game_handler = game_handler
        self.qnetwork = qnetwork
        self.replay_memory = replay_memory
        self.saver = saver
        self.monitor = monitor
        self.train_freq = train_freq
        self.test_mode = test_mode
        self.batch_size = batch_size
        self.save_every_x_episodes = save_every_x_episodes

        self.curr_screen = game_handler.getProcessedImage()

        self.steps = 0
        self.episode_count = 0

    # if we choosing random action, we may not want to calculate action value from network to save time.
    # force_results forces calculating action value for debugging/monitoring
    def _choose_action(self, epsilon, force_results=False):
        rnd = random.random()
        results = None
        if rnd > epsilon or force_results:
            action, results = self.qnetwork.get_best_action(self.replay_memory.last_state())
        if rnd <= epsilon:
            action = random.randrange(self.game_handler.num_actions)
        return action, results

    def _make_action(self, action):
        reward, new_episode = self.game_handler.act(action)
        self.replay_memory.add_experience(self.curr_screen, action, reward, new_episode)
        self.curr_screen = self.game_handler.getProcessedImage()
        return reward, new_episode

    def _train(self, batch_size):
        batch = self.replay_memory.get_batch(batch_size)
        return self.qnetwork.train(batch)

    def populate_replay_memory(self, frames, force_early_stop=False):
        for i in xrange(sys.maxint):
            action, _ = self._choose_action(1)
            _, game_over = self._make_action(action)
            if i >= frames and (force_early_stop or game_over):
                break
            if game_over:
                self.game_handler.reset_game()

    def play_game(self, epsilon, final_eps, eps_diff):
        self.game_handler.reset_game()
        total_reward = 0
        total_steps = 0

        if self.test_mode:
            self.monitor.add_ignored_action('choose_action')
            self.monitor.add_ignored_action('make_action')
            self.monitor.add_ignored_action('train')

        while True:
            total_steps += 1
            self.monitor.report_action_start('choose_action')
            action, results = self._choose_action(epsilon, force_results=True)
            self.monitor.report_action_finish('choose_action', 100)

            self.monitor.report_action_start('make_action')
            reward, game_over = self._make_action(action)
            total_reward += reward
            self.monitor.report_action_finish('make_action', 100)

            if not self.test_mode and total_steps % self.train_freq == 0:
                self.monitor.report_action_start('train')
                error = self._train(self.batch_size)
                self.monitor.report_action_finish('train', 100)
                self.monitor.train_step(epsilon, results, error)
            if self.test_mode:
                self.monitor.test_mode_step(results)

            epsilon = max(final_eps, epsilon - eps_diff)
            if game_over:
                self.episode_count += 1

                self.monitor.episode_finished(total_reward)
                if self.saver is not None and self.episode_count % self.save_every_x_episodes == 0:
                    self.saver.save(global_step=(self.monitor.steps_count + self.saver.get_start_frame()))
                return epsilon

    def play(self, episodes_limit=None, train_steps_limit=None, start_eps=None, final_eps=None,
             exploring_duration=None):
        if episodes_limit is None and train_steps_limit is None:
            raise "episodes and frames limits are None. At least one should be positive integer"
        if episodes_limit is None:
            episodes_limit = sys.maxint
        if train_steps_limit is None:
            train_steps_limit = sys.maxint

        epsilon = start_eps
        eps_diff = (start_eps - final_eps) / exploring_duration
        self.monitor.start_timer()

        for episode_number in xrange(0, episodes_limit):
            epsilon = self.play_game(epsilon, final_eps, eps_diff)
            if self.monitor.steps_count >= train_steps_limit:
                break

    def play_in_test_mode(self, epsilon):
        self.monitor.start_timer()
        while True:
            self.play_game(epsilon, epsilon, 0)
