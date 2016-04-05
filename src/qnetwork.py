import numpy as np
import tensorflow as tf

from monitoring import Monitoring


def prepare_batch(batch):
    if 'prepared' in batch:
        return
    batch['states_1'] = batch['states_1'].transpose([0, 2, 3, 1])
    batch['states_2'] = batch['states_2'].transpose([0, 2, 3, 1])
    batch['prepared'] = True


class Net():
    def __init__(self, output, params):
        self.output = output
        self.params = params


def get_inference(inp, num_actions, name=''):
    with tf.variable_scope(name):
        W1 = tf.get_variable('W1', [8, 8, 4, 16], initializer=tf.truncated_normal_initializer(0, 0.01))
        b1 = tf.get_variable('b1', [16], initializer=tf.constant_initializer(0.01))
        W2 = tf.get_variable('W2', [4, 4, 16, 32], initializer=tf.truncated_normal_initializer(0, 0.01))
        b2 = tf.get_variable('b2', [32], initializer=tf.constant_initializer(0.01))
        W3 = tf.get_variable('W3', [9 * 9 * 32, 256], initializer=tf.truncated_normal_initializer(0, 0.01))
        b3 = tf.get_variable('b3', [256], initializer=tf.constant_initializer(0.01))
        W4 = tf.get_variable('W4', [256, num_actions], initializer=tf.truncated_normal_initializer(0, 0.01))
        b4 = tf.get_variable('b4', [num_actions], initializer=tf.constant_initializer(0.01))

        conv1 = tf.nn.relu(tf.nn.conv2d(inp, W1, [1, 4, 4, 1], 'VALID') + b1)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, [1, 2, 2, 1], 'VALID') + b2)
        reshaped = tf.reshape(conv2, [-1, 9 * 9 * 32])
        fc1 = tf.nn.relu(tf.matmul(reshaped, W3) + b3)
        out = tf.matmul(fc1, W4) + b4

        res = Net(out, [W1, b1, W2, b2, W3, b3, W4, b4])
    return res


def get_inference2(inp, num_actions, name=''):
    with tf.variable_scope(name):
        W1 = tf.get_variable('W1', [8, 8, 4, 32], initializer=tf.truncated_normal_initializer(0, 0.01))
        b1 = tf.get_variable('b1', [32], initializer=tf.constant_initializer(0.01))
        W2 = tf.get_variable('W2', [4, 4, 32, 64], initializer=tf.truncated_normal_initializer(0, 0.01))
        b2 = tf.get_variable('b2', [64], initializer=tf.constant_initializer(0.01))
        W3 = tf.get_variable('W3', [3, 3, 64, 64], initializer=tf.truncated_normal_initializer(0, 0.01))
        b3 = tf.get_variable('b3', [64], initializer=tf.constant_initializer(0.01))
        W4 = tf.get_variable('W4', [7 * 7 * 64, 512], initializer=tf.truncated_normal_initializer(0, 0.01))
        b4 = tf.get_variable('b4', [512], initializer=tf.constant_initializer(0.01))
        W5 = tf.get_variable('W5', [512, num_actions], initializer=tf.truncated_normal_initializer(0, 0.01))
        b5 = tf.get_variable('b5', [num_actions], initializer=tf.constant_initializer(0.01))


        conv1 = tf.nn.relu(tf.nn.conv2d(inp, W1, [1, 4, 4, 1], 'VALID') + b1)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, [1, 2, 2, 1], 'VALID') + b2)
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W3, [1, 1, 1, 1], 'VALID') + b3)
        reshaped = tf.reshape(conv3, [-1, 7 * 7 * 64])
        fc1 = tf.nn.relu(tf.matmul(reshaped, W4) + b4)
        out = tf.matmul(fc1, W5) + b5

        res = Net(out, [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5])
    return res


class DeepQNetwork(object):
    def __init__(self, height, width, session, num_actions, state_frames, gamma, net_type=1,
                 optimizer=tf.train.AdamOptimizer(1e-6)):
        self.height = height
        self.width = width
        self.session = session
        self.num_actions = num_actions
        self.state_frames = state_frames
        self.gamma = gamma
        self.opt = optimizer
        self.train_counter = 0

        self.input_pl = tf.placeholder(tf.float32, [None, height, width, state_frames])
        self.target_pl = tf.placeholder(tf.float32, [None])
        self.actions_pl = tf.placeholder(tf.float32, [None, num_actions])

        if net_type == 1:
            self.network = get_inference(self.input_pl, num_actions, 'main_net')
        else:
            self.network = get_inference2(self.input_pl, num_actions, 'main_net')

        self.prediction = tf.reduce_sum(tf.mul(self.network.output, self.actions_pl), 1)
        #self.error = tf.reduce_mean(tf.square(self.target_pl - self.prediction))
        self.error = tf.square(self.target_pl - self.prediction)

        self.train_step = self.opt.minimize(self.error)

        self.session.run([tf.initialize_all_variables()])

        #todo: throw that away
        self.monitor = Monitoring()

    def get_best_action(self, example):
        vals = self.evaluate_network([example.transpose([1, 2, 0])])[0]
        return np.argmax(vals), vals

    def train(self, batch):
        self.train_counter += 1
        prepare_batch(batch)
        batch_size = batch['batch_size']
        self.monitor.report_action_start('get_batch')
        target = self.get_target(batch)
        self.monitor.report_action_finish('get_batch')
        actions = np.zeros([batch_size, self.num_actions])
        actions[:, list(batch['actions'])] = 1
        self.monitor.report_action_start('pred')
        _, error_val, pred = self.session.run(
            [self.train_step, self.error, self.prediction],
            feed_dict={self.target_pl: target, self.actions_pl: actions, self.input_pl: batch['states_1']})
        self.monitor.report_action_finish('pred')
        return error_val.mean()

    def get_target(self, batch):
        prepare_batch(batch)
        batch_size = batch['batch_size']
        q_vals = self.evaluate_network(batch['states_2'])
        best_actions = np.argmax(q_vals, 1)
        target = np.choose(best_actions, q_vals.T)
        for i in xrange(batch_size):
            if batch['terminations'][i]:
                mul = 0
            else:
                mul = self.gamma
            target[i] = batch['rewards'][i] + mul * target[i]
        return target

    def evaluate_network(self, states_batch):
        return self.session.run(self.network.output, feed_dict={self.input_pl: states_batch})


class DualDeepQNetwork(DeepQNetwork):
    def __init__(self, height, width, session, num_actions, state_frames, gamma, target_net_refresh_rate, net_type=1,
                 optimizer=tf.train.AdamOptimizer(1e-6)):
        DeepQNetwork.__init__(self, height, width, session, num_actions, state_frames, gamma, net_type, optimizer)

        self.target_net_refresh_rate = target_net_refresh_rate
        if net_type == 1:
            self.target_network = get_inference(self.input_pl, num_actions, 'target_net')
        else:
            self.target_network = get_inference2(self.input_pl, num_actions, 'target_net')

    def train(self, batch):
        prepare_batch(batch)
        if self.train_counter % self.target_net_refresh_rate == 0:
            self.update_target()
        return DeepQNetwork.train(self, batch)

    def get_target(self, batch):
        prepare_batch(batch)
        batch_size = batch['batch_size']
        self.monitor.report_action_start('inner_q')
        inner_q = self.evaluate_network(batch['states_2'])
        self.monitor.report_action_finish('inner_q')
        best_actions = np.argmax(inner_q, 1)
        self.monitor.report_action_start('outer_q')
        outer_q = self.evaluate_target(batch['states_2'])
        self.monitor.report_action_finish('outer_q')
        target = np.choose(best_actions, outer_q.T)
        for i in xrange(batch_size):
            if batch['terminations'][i]:
                mul = 0
            else:
                mul = self.gamma
            target[i] = batch['rewards'][i] + mul * target[i]
        return target

    def evaluate_target(self, states_batch):
        return self.session.run(self.target_network.output, feed_dict={self.input_pl: states_batch})

    def update_target(self):
        self._copy_params(self.target_network, self.network)

    def _copy_params(self, target, source):
        for i in range(len(target.params)):
            self.session.run([target.params[i].assign(source.params[i])])
