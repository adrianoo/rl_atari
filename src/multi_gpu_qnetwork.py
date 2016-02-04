import sys
sys.path.insert(0, '../../cnn/cnn/')
sys.path.insert(0, '../../cnn/')

import cnn

import numpy as np
import tensorflow as tf
from utils import average_gradients, get_feed_dict
from qnetwork import get_inference


class MultiGPUDualDeepQNetwork(object):
    def __init__(self, height, width, session, num_actions, state_frames, gamma, gpus=[0],
                 optimizer=tf.train.AdamOptimizer(1e-6)):
        self.height = height
        self.width = width
        self.session = session
        self.num_actions = num_actions
        self.state_frames = state_frames
        self.gamma = gamma
        self.gpus = gpus
        self.opt = optimizer

        self.input_pl_list = [tf.placeholder(tf.float32, [None, height, width, state_frames]) for _ in gpus]
        self.target_pl_list = [tf.placeholder(tf.float32, [None]) for _ in gpus]
        self.actions_pl_list = [tf.placeholder(tf.float32, [None, num_actions]) for _ in gpus]

        gradients = []
        outputs = []
        target_outputs = []
        errors = []
        for i, gpu_nr in enumerate(gpus):
            gpu_name = '/gpu:%d' % gpu_nr
            tower_name = 'tower_%d' % gpu_nr
            with tf.device(gpu_name), tf.name_scope(tower_name):
                network = get_inference(self.input_pl_list[i], num_actions, 'main_net')
                target_network = get_inference(self.input_pl_list[i], num_actions, 'target_net')

                tf.get_variable_scope().reuse_variables()

                prediction = tf.reduce_sum(tf.mul(network.output, self.actions_pl_list[i]), 1)
                error = tf.reduce_mean(tf.square(self.target_pl_list[i] - prediction))

                gradients.append(self.opt.compute_gradients(error))
                outputs.append(network.output)
                target_outputs.append(target_network.output)
                errors.append(error)

        self.grads = average_gradients(gradients)
        self.out = tf.concat(0, outputs)
        self.target_out = tf.concat(0, target_outputs)
        self.error = tf.reduce_mean(errors)

        self.network = get_inference(self.input_pl_list[0], num_actions, 'main_net')
        self.target_network = get_inference(self.input_pl_list[0], num_actions, 'target_net')
        self.train_step = self.opt.apply_gradients(self.grads)

        self.session.run([tf.initialize_all_variables()])
        self.update_target()

    def get_best_action(self, example):
        vals = self.session.run(self.network.output, feed_dict={self.input_pl_list[0]: [example]})[0]
        return np.argmax(vals), vals

    def train(self, batch):
        batch_size = batch['batch_size']
        target = self.get_target(batch)
        actions = np.zeros([batch_size, self.num_actions])
        actions[:, list(batch['actions'])] = 1
        _, error_val = self.session.run(
            [self.train_step, self.error],
            feed_dict=get_feed_dict([(self.target_pl_list, target), (self.actions_pl_list, actions),
                                     (self.input_pl_list, batch['states_1'])], len(self.gpus))
        )
        return error_val

    def get_target(self, batch):
        # todo: think about target network in multiple gpu setup. Right now its not used.
        """
        batch_size = batch['batch_size']
        inner_q = self.evaluate_network(batch['states_2'])
        best_actions = np.argmax(inner_q, 1)
        outer_q = self.evaluate_target(batch['states_2'])
        target = np.choose(best_actions, outer_q.T)
        """
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
        return self.session.run(self.out, feed_dict=get_feed_dict([(self.input_pl_list, batch)], len(self.gpus)))

    def evaluate_target(self, batch):
        return self.session.run(self.target_out, feed_dict=get_feed_dict([(self.input_pl_list, batch)], len(self.gpus)))

    # todo: think about target network in multiple gpu setup. Right now its not used.
    def update_target(self):
        self._copy_params(self.target_network, self.network)

    def _copy_params(self, target, source):
        for i in range(len(target.params)):
            self.session.run([target.params[i].assign(source.params[i])])
