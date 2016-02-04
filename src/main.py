#!/usr/bin/env python

import sys
import random


import logging
import tensorflow as tf

from agent import Agent
from qnetwork import DualDeepQNetwork
from multi_gpu_qnetwork import MultiGPUDualDeepQNetwork
from image_processing import crop_and_resize
from game_handler import GameStateHandler
from replay_memory import ReplayMemoryManager
from saver import Saver

ROM_FILE = '../roms/breakout.bin'

DISCOUNT_FACTOR = 0.99
START_EPSILON = 1.0
FINAL_EPSILON = 0.05
EXPLORING_DURATION = 2000
LEARNING_BEYOND_EXPLORING = 2000
REPLAY_MEMORY_SIZE = 1000000
MIN_REPLAY_MEMORY = 10000
BATCH_SIZE = 32
EMULATOR_FRAME_SKIP = 1
FRAME_SKIP = 4
STATE_FRAMES = 4
TARGET_NET_REFRESH_RATE = 10000 * FRAME_SKIP
IMAGE_HEIGHT = 84
IMAGE_WIDTH = 84
DATA_DIR = '../data/networks1/'
LOG_FILE = DATA_DIR + 'logs'

LEARNING_AGENT_STATE = 'learning'
GUESSING_AGENT_STATE = 'guessing'


def play():
    with tf.Session() as sess:
        logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler())

        game_handler = GameStateHandler(frame_skip=EMULATOR_FRAME_SKIP, random_seed=123, state_frames=STATE_FRAMES, use_sdl=False,
                                        image_processing=lambda x: crop_and_resize(x, IMAGE_HEIGHT, IMAGE_WIDTH))
        game_handler.loadROM(ROM_FILE)

        height, width = game_handler.getScreenDims()
        logging.info('Screen resolution is %dx%d' % (height, width))
        num_actions = game_handler.num_actions
        optimizer = tf.train.RMSPropOptimizer(0.00025, 0.9, 0.95, 0.01)
#        qnetwork = DualDeepQNetwork(IMAGE_HEIGHT, IMAGE_WIDTH, sess, num_actions, STATE_FRAMES, DISCOUNT_FACTOR,
#                                    optimizer=optimizer)
        qnetwork = MultiGPUDualDeepQNetwork(IMAGE_HEIGHT, IMAGE_WIDTH, sess, num_actions, STATE_FRAMES, DISCOUNT_FACTOR,
                                            optimizer=optimizer, gpus=[0, 1, 2, 3])
        saver = Saver(sess, DATA_DIR)
        if saver.replay_memory_found():
            replay_memory = saver.get_replay_memory()
        else:
            replay_memory = ReplayMemoryManager(IMAGE_HEIGHT, IMAGE_WIDTH, REPLAY_MEMORY_SIZE)
        agent = Agent(game_handler, qnetwork, replay_memory, saver)

        sess.run(tf.initialize_all_variables())
        saver.restore(DATA_DIR)

        start_epsilon = max(FINAL_EPSILON,
                            START_EPSILON - saver.get_start_frame() * (START_EPSILON - FINAL_EPSILON) / EXPLORING_DURATION)
        agent.populate_replay_memory(MIN_REPLAY_MEMORY)
        agent.play(frames_limit=LEARNING_BEYOND_EXPLORING+EXPLORING_DURATION, start_eps=start_epsilon,
                   final_eps=FINAL_EPSILON, exploring_duration=EXPLORING_DURATION - saver.get_start_frame())


def main(argv=None):
    play()


if __name__ == '__main__':
    tf.app.run()
