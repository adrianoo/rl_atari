import argparse
import logging
import tensorflow as tf

import qnetwork
import multi_gpu_qnetwork

from agent import Agent
from image_processing import crop_and_resize
from game_handler import GameStateHandler
from monitoring import Monitoring
from replay_memory import ReplayMemoryManager
from saver import Saver


parser = argparse.ArgumentParser()

# environment
parser.add_argument('--rom_name', type=str, default='breakout.bin')
parser.add_argument('--rom_directory', type=str, default='../roms/')
parser.add_argument('--frame_skip', type=int, default=4)
parser.add_argument('--repeat_action_probability', type=float, default=0.0)
parser.add_argument('--use_sdl', dest='use_sdl', action='store_true', default=False)
parser.add_argument('--random_seed', type=int, default=666)
parser.add_argument('--full_action_set', dest='minimum_action_set', action='store_false', default=True)
parser.add_argument('--cut_top', type=int, default=40)

# replay_memory
parser.add_argument('--replay_memory_size', type=int, default=1000000)
parser.add_argument('--min_replay_memory', type=int, default=50000)
# todo: Not yet supported
parser.add_argument('--prioritized_replay_memory', dest='prioritized_replay_memory', action='store_true', default=False)
parser.add_argument('--reward_clip_min', type=float, default=-1.0)
parser.add_argument('--reward_clip_max', type=float, default=1.0)

# batch
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--state_frames', type=int, default=4)

# agent:
parser.add_argument('--train_freq', type=int, default=1) # originally 4, but I can't see reasons why it should be 4
parser.add_argument('--discount_factor', type=float, default=0.99)
parser.add_argument('--number_of_train_steps', type=int, default=10000000, help='Number of train steps including\
                    exploration period. Actual number of train steps may be lower due to train_freq > 1')
# todo: use this in launcher (its implemented in agent)
parser.add_argument('--number_of_episodes', type=int, default=None)

# exploration
parser.add_argument('--start_epsilon', type=float, default=1.0)
parser.add_argument('--final_epsilon', type=float, default=0.05)
parser.add_argument('--exploration_duration', type=int, default=1000000)

# dqn
parser.add_argument('--optimizer', choices=['rmsprop'], default='rmsprop')
parser.add_argument('--learning_rate', type=float, default=0.00025)
parser.add_argument('--decay', type=float, default=0.95)
parser.add_argument('--rmsprop_epsilon', type=float, default=0.01)
parser.add_argument('--double_dqn', dest='double_dqn', action='store_true', default=False)
parser.add_argument('--target_net_refresh_rate', type=int, default=10000)
parser.add_argument('--net_type', type=int, default=1)

# other
parser.add_argument('--test_mode', dest='test_mode', action='store_true', default=False)
parser.add_argument('--epsilon_in_test_mode', type=float, default=0.0)
# todo: currently may not work properly - network saving not working. Also add multi gpus numbers as param
parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true', default=False)
# todo: change this path. Also create non existing directiories
parser.add_argument('--data_dir', type=str, default='../data/networks1')
parser.add_argument('--log_file', type=str, default='../data/networks1/logs')
parser.add_argument('--saving_freq', type=int, default=100, help='Save network after this many episodes')

args = parser.parse_args()
# todo: customization of neural net?
args.image_height = 84
args.image_width = 84


def main(_=None):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        logging.basicConfig(filename=args.log_file, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler())

        game_handler = GameStateHandler(
                args.rom_directory + args.rom_name,
                random_seed=args.random_seed,
                frame_skip=args.frame_skip,
                use_sdl=args.use_sdl,
                repeat_action_probability=args.repeat_action_probability,
                minimum_actions=args.minimum_action_set,
                test_mode=args.test_mode,
                image_processing=lambda x: crop_and_resize(x, args.image_height, args.image_width, args.cut_top))
        num_actions = game_handler.num_actions

        if args.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=args.learning_rate,
                    decay=args.decay,
                    momentum=0.0,
                    epsilon=args.rmsprop_epsilon)

        if not args.multi_gpu:
            if args.double_dqn:
                net = qnetwork.DualDeepQNetwork(args.image_height, args.image_width, sess, num_actions,
                                                args.state_frames, args.discount_factor, args.target_net_refresh_rate,
                                                net_type=args.net_type, optimizer=optimizer)
            else:
                net = qnetwork.DeepQNetwork(args.image_height, args.image_width, sess, num_actions, args.state_frames,
                                            args.discount_factor, net_type=args.net_type, optimizer=optimizer)
        else:
            net = multi_gpu_qnetwork.MultiGPUDualDeepQNetwork(args.image_height, args.image_width, sess, num_actions,
                                                              args.state_frames, args.discount_factor,
                                                              optimizer=optimizer, gpus=[0, 1, 2, 3])

        saver = Saver(sess, args.data_dir)
        if saver.replay_memory_found():
            replay_memory = saver.get_replay_memory()
        else:
            if args.test_mode:
                logging.error('NO SAVED NETWORKS IN TEST MODE!!!')
            replay_memory = ReplayMemoryManager(args.image_height, args.image_width, args.state_frames,
                                                args.replay_memory_size, reward_clip_min=args.reward_clip_min,
                                                reward_clip_max=args.reward_clip_max)

        # todo: add parameters to handle monitor
        monitor = Monitoring(log_train_step_every=100, smooth_episode_scores_over=50)

        agent = Agent(
                game_handler=game_handler,
                qnetwork=net,
                replay_memory=replay_memory,
                saver=saver,
                monitor=monitor,
                train_freq=args.train_freq,
                test_mode=args.test_mode,
                batch_size=args.batch_size,
                save_every_x_episodes=args.saving_freq)

        sess.run(tf.initialize_all_variables())
        saver.restore(args.data_dir)
        start_epsilon = max(args.final_epsilon,
                            args.start_epsilon - saver.get_start_frame() * (args.start_epsilon - args.final_epsilon) / args.exploration_duration)
        exploring_duration = max(args.exploration_duration - saver.get_start_frame(), 1)

        if args.test_mode:
            agent.populate_replay_memory(args.state_frames, force_early_stop=True)
            agent.play_in_test_mode(args.epsilon_in_test_mode)
        else:
            agent.populate_replay_memory(args.min_replay_memory)
            agent.play(train_steps_limit=args.number_of_train_steps, start_eps=start_epsilon,
                       final_eps=args.final_epsilon, exploring_duration=exploring_duration)


if __name__ == '__main__':
#    tf.app.run()
    main()
