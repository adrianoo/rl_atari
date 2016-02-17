import logging

from agent import Agent
from image_processing import crop_and_resize
from game_handler import GameStateHandler
from monitoring import Monitoring
from replay_memory import ReplayMemoryManager


import theano_qnetwork

def run(args):
    logging.basicConfig(filename=args.LOG_FILE, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    game_handler = GameStateHandler(random_seed=123, frame_skip=args.FRAME_SKIP, use_sdl=False,
                                    image_processing=lambda x: crop_and_resize(x, args.IMAGE_HEIGHT, args.IMAGE_WIDTH))
    game_handler.loadROM(args.ROM_FILE)

    height, width = game_handler.getScreenDims()
    logging.info('Screen resolution is %dx%d' % (height, width))
    num_actions = game_handler.num_actions

    net = theano_qnetwork.DeepQNetwork(args.IMAGE_HEIGHT, args.IMAGE_WIDTH, num_actions, args.STATE_FRAMES, args.DISCOUNT_FACTOR)

    replay_memory = ReplayMemoryManager(args.IMAGE_HEIGHT, args.IMAGE_WIDTH, args.STATE_FRAMES, args.REPLAY_MEMORY_SIZE)

    monitor = Monitoring(log_train_step_every=100, smooth_episode_scores_over=50)
    agent = Agent(game_handler, net, replay_memory, None, monitor, args.TRAIN_FREQ, batch_size=args.BATCH_SIZE)

    start_epsilon = args.START_EPSILON
    exploring_duration = args.EXPLORING_DURATION

    agent.populate_replay_memory(args.MIN_REPLAY_MEMORY)
    agent.play(train_steps_limit=args.LEARNING_BEYOND_EXPLORING+args.EXPLORING_DURATION, start_eps=start_epsilon,
               final_eps=args.FINAL_EPSILON, exploring_duration=exploring_duration)
