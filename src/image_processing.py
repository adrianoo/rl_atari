import cv2
import tensorflow as tf


def crop_and_resize(image, height=84, width=84):
    return cv2.resize(image[40:, :], (height, width))


#obsolete probably
class TFImageProcessor(object):
    def __init__(self, sess, image_height, image_width, height=84, width=84):
        self.sess = sess
        self.image_height = image_height
        self.image_width = image_width
        self.height = height
        self.width = width
        self.image_pl = tf.placeholder(tf.int8, [self.image_height, self.image_width, 1])

#    def crop(self):
#        return tf.image.crop_to_bounding_box(self.image_pl, 40, 0, 210, 160)

    def algo1(self):
        return tf.image.resize_images(self.image_pl, self.height, self.width)

    def process(self, image):
        return self.algo1().eval(feed_dict={self.image_pl: image}, session=self.sess).squeeze(2) / 128.0

    def process2(self, image):
        return tf.image.crop_to_bounding_box(self.image_pl)
