from Queue import deque
import logging
import time


class Monitoring(object):
    def __init__(self, log_train_step_every=10, smooth_episode_scores_over=10):
        self.steps_count = 0
        self.episodes_count = 0
        self.timer_started = False
        self.episode_start = 0
        self.steps_deq = deque()
        self.episodes_deq = deque()
        self.steps_window = log_train_step_every
        self.episodes_window = smooth_episode_scores_over

        self.deqs = dict()
        self.starts = dict()
        self.steps = dict()
        self.ignored = set()
        pass

    def start_timer(self):
        self.timer_started = True
        now = time.time()
        self.steps_deq.append(now)
        self.episode_start = now

    def train_step(self, epsilon, results, error):
        assert self.timer_started
        self.steps_count += 1
        self.steps_deq.append(time.time())
        if len(self.steps_deq) > self.steps_window + 1:
            self.steps_deq.popleft()
        if self.steps_count % self.steps_window == 0:
            sec_per_train = (self.steps_deq[-1] - self.steps_deq[0]) / self.steps_window
            logging.info('Step %d; sec per step %.4f; steps per sec %.2f; epsilon %.4f; error %.8f; results %s' %
                         (self.steps_count, sec_per_train, 1./sec_per_train, epsilon, error, str(results)))
            print results

    def test_mode_step(self, results):
        logging.info('q-values %s' % str(results))

    def episode_finished(self, score):
        assert self.timer_started
        self.episodes_count += 1
        self.episodes_deq.append(score)
        if len(self.episodes_deq) > self.episodes_window:
            self.episodes_deq.popleft()
        logging.info('Episode %d; average score over last %d games: %.2f\n\n' %
                     (self.episodes_count, len(self.episodes_deq),
                      float(sum(self.episodes_deq)) / len(self.episodes_deq)))

    def add_ignored_action(self, name):
        self.ignored.add(name)

    def report_action_start(self, name):
        if name in self.ignored:
            return
        self.starts[name] = time.time()

    def report_action_finish(self, name, every_k=100):
        if name in self.ignored:
            return
        assert name in self.starts
        duration = time.time() - self.starts[name]
        if not name in self.deqs:
            self.deqs[name] = deque()
            self.steps[name] = 0
        self.deqs[name].append(duration)
        self.steps[name] += 1

        if len(self.deqs[name]) > self.steps_window + 1:
            self.deqs[name].popleft()
        if self.steps[name] % every_k == 0:
            sec_per_action = float(sum(self.deqs[name])) / every_k
            logging.info('Action %s over %d last occurences took %.4f on average' %
                         (name + ' ' * (20 - len(name)), every_k, sec_per_action))


