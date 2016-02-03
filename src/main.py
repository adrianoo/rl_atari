#!/usr/bin/env python

import sys
import random


import numpy as np
import tensorflow as tf

from agent import *
from inference import *
from image_processing import *
from game_handler import *
from replay_memory import *

ROM_FILE = '../roms/breakout.bin'
MAX_ITERATIONS = 15000000

DISCOUNT_FACTOR = 0.99
START_EPSILON = 1.0
FINAL_EPSILON = 0.05
EXPLORING_DURATION = 20000
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY = 50000
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
    sess = tf.Session()
    game_handler = GameStateHandler(frame_skip=EMULATOR_FRAME_SKIP, random_seed=123, state_frames=STATE_FRAMES, use_sdl=False,
                                    image_processing=lambda x: crop_and_resize(x, IMAGE_HEIGHT, IMAGE_WIDTH))
    game_handler.loadROM(ROM_FILE)

    height, width = game_handler.getScreenDims()
    image_processor = TFImageProcessor(sess, height, width)
    num_actions = game_handler.num_actions
    net_manager = DualDeepQNetwork(IMAGE_HEIGHT, IMAGE_WIDTH, sess, num_actions, STATE_FRAMES, DISCOUNT_FACTOR)
    replay_memory = ReplayMemoryManager(IMAGE_HEIGHT, IMAGE_WIDTH, REPLAY_MEMORY_SIZE)
    agent = Agent(game_handler, net_manager, replay_memory)

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    checkpoint = tf.train.get_checkpoint_state(DATA_DIR)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Loaded %s" % checkpoint.model_checkpoint_path
        start_frame = int(checkpoint.model_checkpoint_path[checkpoint.model_checkpoint_path.rfind('-') + 1:])
    else:
        print "Nothing loaded!"
        start_frame = 0

    new_episode = True
    episode_count = 0
    epsilon = START_EPSILON
    agent_state = GUESSING_AGENT_STATE
    curr_screen = game_handler.getProcessedImage()
    for i in xrange(1, MAX_ITERATIONS+1):
        if new_episode:
            episode_count += 1
            total_reward = 0
        if len(replay_memory) >= MIN_REPLAY_MEMORY:
            agent_state = LEARNING_AGENT_STATE

        if i % 4 == 1:
            if agent_state == LEARNING_AGENT_STATE:
                epsilon = max(epsilon - (START_EPSILON - FINAL_EPSILON) / EXPLORING_DURATION, FINAL_EPSILON)
                action, results = net_manager.get_best_action(replay_memory.last_state())
            if random.random() <= epsilon or agent_state == GUESSING_AGENT_STATE:
                action = random.randrange(game_handler.num_actions)

        curr_lives = game_handler.lives()
        reward = game_handler.act(action)
        new_episode = game_handler.game_over() or game_handler.lives() < curr_lives
        replay_memory.add_experience(curr_screen, action, reward, new_episode)
        curr_screen = game_handler.getProcessedImage()
        total_reward += reward

        if agent_state == LEARNING_AGENT_STATE and i % 4 == 0:
            batch = replay_memory.get_batch(BATCH_SIZE)
            error_val = net_manager.train(batch)
        else:
            error_val = -1

        if i % TARGET_NET_REFRESH_RATE == 0:
            net_manager.update_target()

        #if reward > 0:
            #game_handler.saveProcessedImage(
            #        DATA_DIR + 'screens/screen_%d' % (game_handler.getFrameNumber() + start_frame))
        if i % 100 == 0:
            if agent_state == GUESSING_AGENT_STATE:
                log('Frame %d; state %s\t' % \
                    (game_handler.getFrameNumber() + start_frame, agent_state), LOG_FILE)
            elif agent_state == LEARNING_AGENT_STATE:
                log('Frame %d; state %s\tresults %s\tEPSILON %.5f\terror %.10f' % \
                    (game_handler.getFrameNumber() + start_frame, agent_state, str(results), epsilon, error_val),
                    LOG_FILE)

        if new_episode:
            log('Episode %d passed after %d frames, with total reward of %d\n\n\n' % \
                (episode_count, game_handler.getEpisodeFrameNumber(), total_reward), LOG_FILE)
            game_handler.reset_game()
            saver.save(sess, DATA_DIR + 'net', global_step=(game_handler.getFrameNumber() + start_frame))


def main():
    play()

if __name__ == '__main__':
    main()
