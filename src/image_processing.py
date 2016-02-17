import cv2


def crop_and_resize(image, height=84, width=84, cut_top=40):
    return cv2.resize(image[cut_top:, :], (height, width))
