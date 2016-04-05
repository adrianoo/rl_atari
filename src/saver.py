import logging
import os
import tensorflow as tf


class Saver(tf.train.Saver):
    def __init__(self, session, dir, continue_training, *args, **kwargs):
        tf.train.Saver.__init__(self, *args, **kwargs)
        self.session = session
        self.dir = dir
        self.continue_training = continue_training
        self.start_frame = 0

    def save(self, *args, **kwargs):
        tf.train.Saver.save(self, self.session, self.dir, *args, **kwargs)

    def restore(self, dir):
        d = os.path.dirname(dir)
        if not os.path.exists(d):
            os.makedirs(d)
        checkpoint = tf.train.get_checkpoint_state(dir)
        if self.continue_training and checkpoint and checkpoint.model_checkpoint_path:
            tf.train.Saver.restore(self, self.session, checkpoint.model_checkpoint_path)
            logging.info("Saver::Variables loaded from %s" % checkpoint.model_checkpoint_path)
            self.start_frame = int(checkpoint.model_checkpoint_path[checkpoint.model_checkpoint_path.rfind('-') + 1:])
        else:
            logging.info("Saver::No variables loaded!")

    def get_start_frame(self):
        return self.start_frame

    # todo: handle saving/restoring replay memory (dont forget about self.continue_training)
    def replay_memory_found(self):
        return False

    def get_replay_memory(self):
        return None
