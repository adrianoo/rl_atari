import logging
import random
import sys


class Agent(object):
    def __init__(self, game_handler, qnetwork, replay_memory, saver,
                 batch_size=32, start_epsilon=1, final_epsilon=0.05, exploring_duration=1000000,
                 frame_skip=4, target_net_refresh_rate=None):
        self.game_handler = game_handler
        self.qnetwork = qnetwork
        self.replay_memory = replay_memory
        self.saver = saver
        self.batch_size = batch_size
        self.start_epsilon = start_epsilon
        self.final_epsilon = final_epsilon
        self.exploring_duration = exploring_duration
        self.frame_skip = frame_skip
        if target_net_refresh_rate is None:
            self.target_net_refresh_rate = 10000 * self.frame_skip
        else:
            self.target_net_refresh_rate = target_net_refresh_rate

        self.curr_screen = game_handler.getProcessedImage()
        self.total_reward_curr_episode = 0

    # if we choosing random action, we may not want to calculate action value from network to save time.
    # force_results forces calculating action value for debugging/monitoring
    def _choose_action(self, epsilon, force_results=False):
        results = None
        if force_results:
            action, results = self.qnetwork.get_best_action(self.replay_memory.last_state())
        if random.random() <= epsilon:
            action = random.randrange(self.game_handler.num_actions)
        return action, results

    def _make_action(self, action, times=1):
        for i in xrange(times):
            curr_lives = self.game_handler.lives()
            reward = self.game_handler.act(action)
            new_episode = self.game_handler.game_over() or self.game_handler.lives() < curr_lives
            self.replay_memory.add_experience(self.curr_screen, action, reward, new_episode)
            self.curr_screen = self.game_handler.getProcessedImage()
            self.total_reward_curr_episode += reward
            if self.game_handler.getFrameNumber() % self.target_net_refresh_rate == 0:
                self.qnetwork.update_target()
            if new_episode:
                break
        return new_episode

    def _train(self, batch_size):
        batch = self.replay_memory.get_batch(batch_size)
        return self.qnetwork.train(batch)

    def populate_replay_memory(self, frames):
        for i in xrange(sys.maxint):
            action, _ = self._choose_action(1)
            game_over = self._make_action(action)
            if i >= frames and game_over:
                break
            if game_over:
                self.game_handler.reset_game()

    def play(self, episodes_limit=None, frames_limit=None, start_eps=None, final_eps=None, exploring_duration=None):
        if episodes_limit is None and frames_limit is None:
            raise "episodes and frames limits are None. At least one should be positive integer"
        if episodes_limit is None:
            episodes_limit = sys.maxint
        if frames_limit is None:
            frames_limit = sys.maxint
        if start_eps is None:
            start_eps = self.start_epsilon
        if final_eps is None:
            final_eps = self.final_epsilon
        if exploring_duration is None:
            exploring_duration = self.exploring_duration

        epsilon = start_eps
        logging.debug('start_frame = %d; getFrameNumber = %d' % (self.saver.get_start_frame(),
                                                                 self.game_handler.getFrameNumber()))
        first_frame_number = self.game_handler.getFrameNumber() + self.saver.get_start_frame()

        for episode_number in xrange(0, episodes_limit):
            self.game_handler.reset_game()
            self.total_reward_curr_episode = 0
            while True:
                action, results = self._choose_action(epsilon, force_results=True)
                game_over = self._make_action(action, self.frame_skip)
                error = self._train(self.batch_size)

                msg = 'Frame %d; epsilon %.4f; error %.5f' % (
                    self.game_handler.getFrameNumber() + self.saver.get_start_frame(),
                    epsilon,
                    error
                )
                if results is not None:
                    msg += '; results %s' % str(results)
                logging.info(msg)

                epsilon = max(final_eps, epsilon - self.frame_skip * (start_eps - final_eps) / exploring_duration)
                if game_over:
                    logging.info('Episode %d passed after %d frames, with total reward of %d\n\n' % \
                        (episode_number, self.game_handler.getEpisodeFrameNumber(),
                         self.total_reward_curr_episode))
                    self.saver.save(global_step=(self.game_handler.getFrameNumber() + self.saver.get_start_frame()))
                    break
            if self.game_handler.getFrameNumber() + self.saver.get_start_frame() - first_frame_number >= frames_limit:
                break
