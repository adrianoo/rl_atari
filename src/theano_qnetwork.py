import numpy as np
import theano
import theano.tensor as T

import lasagne
import lasagne.layers.dnn

from monitoring import Monitoring


def gen_updates_rmsprop(all_parameters, all_grads, learning_rate=1.0, rho=0.9, epsilon=1e-6, get_diffs=False, what_stats=None):
    """
    epsilon is not included in Hinton's video, but to prevent problems with relus repeatedly having 0 gradients, it is included here.
    Watch this video for more info: http://www.youtube.com/watch?v=O3sxAc4hxZU (formula at 5:20)
    also check http://climin.readthedocs.org/en/latest/rmsprop.html
    """
    all_accumulators = [theano.shared(param_i.get_value() * 0.) for param_i in
                        all_parameters]  # initialise to zeroes with the right shape

    updates = []
    infos = []

    for param_i, grad_i, acc_i in zip(all_parameters, all_grads, all_accumulators):
        acc_i_new = rho * acc_i + (1 - rho) * grad_i ** 2
        updates.append((acc_i, acc_i_new))
        scaled_grad = grad_i / T.sqrt(acc_i_new + epsilon)

        update = -learning_rate * scaled_grad
        updates.append((param_i, param_i + update))

    return updates


def get_inference(num_actions, height, width, state_frames, conv=lasagne.layers.dnn.Conv2DDNNLayer):
    net = lasagne.layers.InputLayer(shape=(None, state_frames, height, width))
    net = conv(
        net, num_filters=32, filter_size=(8, 8), stride=4, pad='valid',
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    net = conv(
        net, num_filters=64, filter_size=(4, 4), stride=2, pad='valid',
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    net = conv(
        net, num_filters=64, filter_size=(3, 3), stride=1, pad='valid',
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    net = lasagne.layers.DenseLayer(
        net, num_units=256, nonlinearity=lasagne.nonlinearities.rectify
    )
    net = lasagne.layers.DenseLayer(
        net, num_units=num_actions, nonlinearity=lasagne.nonlinearities.identity
    )
    return net

class DeepQNetwork(object):
    def __init__(self, height, width, num_actions, state_frames, gamma, batch_size=32):
        self.gamma = gamma
        self.num_actions = num_actions
        self.input_var = T.tensor4('inputs')
        self.target_var = T.vector('targets')
        self.actions_var = T.matrix()

        self.input_shared = theano.shared(np.zeros([batch_size, state_frames, height, width]),
                                          dtype=theano.tensor.floatX)
        self.taret_shared = theano.shared(np.zeros([batch_size, 1]), dtype=theano.tensor.floatX)
        self.actions_shared = theano.shared(np.zeros([batch_size, num_actions]).astype('int32'),
                                            dtype=theano.tensor.int32)

        self.net = get_inference(self.input_var, num_actions, height, width, state_frames)
        self.prediction = (lasagne.layers.get_output(self.net) * self.actions_var).sum(axis=1)
        self.error = T.sqr(self.target_var - self.prediction).mean()

        params = lasagne.layers.get_all_params(self.net, trainable=True)
        #self.opt = lasagne.updates.rmsprop(
        #    self.error, params, learning_rate=0.0025, epsilon=0.01)
        #self.opt = lasagne.updates.adagrad(
        #    self.error, params, learning_rate=0.001
        #)
        self.opt = gen_updates_rmsprop(params, T.grad(self.error, params))

        self.train_step = theano.function([self.input_var, self.target_var, self.actions_var],
                                          self.error, updates=self.opt, allow_input_downcast=True)
        self.net_fun = theano.function([self.input_var], lasagne.layers.get_output(self.net),
                                       allow_input_downcast=True)

        self.monitor = Monitoring()

        self.pred_fun = theano.function([self.input_var, self.actions_var], self.prediction, allow_input_downcast=True)
        self.error_fun = theano.function([self.input_var, self.target_var, self.actions_var], self.error,
                                         allow_input_downcast=True)


    def get_best_action(self, example):
        vals = self.evaluate_network([example])[0]
        return np.argmax(vals), vals

    def evaluate_network(self, batch):
        return self.net_fun(batch)

    def train(self, batch):
        batch_size = batch['batch_size']
        self.monitor.report_action_start('get_target')
        target = self.get_target(batch)
        self.monitor.report_action_finish('get_target')
        actions = np.zeros([batch_size, self.num_actions])
        actions[:, list(batch['actions'])] = 1


        self.monitor.report_action_start('pred_fun')
        self.pred_fun(batch['states_1'], actions)
        self.monitor.report_action_finish('pred_fun')

        self.monitor.report_action_start('error_fun')
        self.error_fun(batch['states_1'], target, actions)
        self.monitor.report_action_finish('error_fun')



        self.monitor.report_action_start('train_step')
        error_val = self.train_step(batch['states_1'], target, actions)
        self.monitor.report_action_finish('train_step')


        return error_val

    def get_target(self, batch):
        batch_size = batch['batch_size']
        self.monitor.report_action_start('only_q')
        q_vals = self.evaluate_network(batch['states_2'])
        self.monitor.report_action_finish('only_q')
        best_actions = np.argmax(q_vals, 1)
        target = np.choose(best_actions, q_vals.T)
        for i in xrange(batch_size):
            if batch['terminations'][i]:
                mul = 0
            else:
                mul = self.gamma
            target[i] = batch['rewards'][i] + mul * target[i]
        return target


def main():
    net = DeepQNetwork(84, 84, 4, 4, 0.95)

if __name__ == '__main__':
    main()
