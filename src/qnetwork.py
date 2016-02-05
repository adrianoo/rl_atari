import sys
sys.path.insert(0, '../../cnn/cnn/')
sys.path.insert(0, '../../cnn/')

import cnn

import numpy as np
import tensorflow as tf


def get_inference(inp, num_actions, name=''):
    with tf.variable_scope(name):
        net = cnn.InputLayer(inp)
        net = cnn.ConvLayer(net, 8, 8, channels=32, stride=4, activation=tf.nn.relu)
        net = cnn.PoolLayer(net)
        net = cnn.ConvLayer(net, 4, 4, channels=64, stride=2, activation=tf.nn.relu)
        net = cnn.PoolLayer(net)
        net = cnn.ConvLayer(net, 3, 3, channels=64, stride=1, activation=tf.nn.relu)
        net = cnn.PoolLayer(net)
        net = cnn.FlattenLayer(net)
        net = cnn.FCLayer(net, 256, activation=tf.nn.relu)
        net = cnn.FCLayer(net, num_actions)
    return net


class DeepQNetwork(object):
    def __init__(self, height, width, session, num_actions, state_frames, gamma,
                 optimizer=tf.train.AdamOptimizer(1e-6)):
        self.height = height
        self.width = width
        self.session = session
        self.num_actions = num_actions
        self.state_frames = state_frames
        self.gamma = gamma
        self.opt = optimizer

        self.input_pl = tf.placeholder(tf.float32, [None, height, width, state_frames])
        self.target_pl = tf.placeholder(tf.float32, [None])
        self.actions_pl = tf.placeholder(tf.float32, [None, num_actions])

        self.network = get_inference(self.input_pl, num_actions, 'main_net')

        self.prediction = tf.reduce_sum(tf.mul(self.network.output, self.actions_pl), 1)
        self.error = tf.reduce_mean(tf.square(self.target_pl - self.prediction))

        self.train_step = self.opt.minimize(self.error)

        self.session.run([tf.initialize_all_variables()])

    def get_best_action(self, example):
        vals = self.evaluate_network([example])[0]
        return np.argmax(vals), vals

    def train(self, batch):
        batch_size = batch['batch_size']
        target = self.get_target(batch)
        actions = np.zeros([batch_size, self.num_actions])
        actions[:, list(batch['actions'])] = 1
        _, error_val, pred = self.session.run(
            [self.train_step, self.error, self.prediction],
            feed_dict={self.target_pl: target, self.actions_pl: actions, self.input_pl: batch['states_1']})
        return error_val

    def get_target(self, batch):
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

    def evaluate_network(self, batch):
        return self.session.run(self.network.output, feed_dict={self.input_pl: batch})


class DualDeepQNetwork(DeepQNetwork):
    def __init__(self, height, width, session, num_actions, state_frames, gamma, target_net_refresh_rate,
                 optimizer=tf.train.AdamOptimizer(1e-6)):
        DeepQNetwork.__init__(self, height, width, session, num_actions, state_frames, gamma, optimizer)

        self.target_net_refresh_rate = target_net_refresh_rate
        self.target_network = get_inference(self.input_pl, num_actions, 'target_net')
        self.train_counter = 0

    def train(self, batch):
        if self.train_counter % self.target_net_refresh_rate == 0:
#            pass
            self.update_target()
        self.train_counter += 1
        return DeepQNetwork.train(self, batch)

    def get_target(self, batch):
        batch_size = batch['batch_size']
        inner_q = self.evaluate_network(batch['states_2'])
        best_actions = np.argmax(inner_q, 1)
        outer_q = self.evaluate_target(batch['states_2'])
        target = np.choose(best_actions, outer_q.T)
        for i in xrange(batch_size):
            if batch['terminations'][i]:
                mul = 0
            else:
                mul = self.gamma
            target[i] = batch['rewards'][i] + mul * target[i]
        return target

    def evaluate_target(self, batch):
        return self.session.run(self.target_network.output, feed_dict={self.input_pl: batch})

    def update_target(self):
        self._copy_params(self.target_network, self.network)

    def _copy_params(self, target, source):
        for i in range(len(target.params)):
            self.session.run([target.params[i].assign(source.params[i])])
