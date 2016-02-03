import sys

sys.path.insert(0, '../../../atari/Arcade-Learning-Environment-0.5.1/')
sys.path.insert(0, '../../../Arcade-Learning-Environment-0.5.1/')

#sys.path.insert(0, '../../atari/Arcade-Learning-Environment-0.5.1/')
#sys.path.insert(0, '../../Arcade-Learning-Environment-0.5.1/')


from collections import deque
import matplotlib.pyplot as plt

from ale_python_interface import ALEInterface


class GameStateHandler(ALEInterface):
    def __init__(self, random_seed=None, frame_skip=2, state_frames=4, minimum_actions=True, use_sdl=False,
                 image_processing=None):
        ALEInterface.__init__(self)

        # Set USE_SDL to true to display the screen. ALE must be compilied
        # with SDL enabled for this to work. On OSX, pygame init is used to
        # proxy-call SDL_main.
        if use_sdl:
            if sys.platform == 'darwin':
                import pygame
                pygame.init()
                self.setBool('sound', False)  # Sound doesn't work on OSX
            elif sys.platform.startswith('linux'):
                self.setBool('sound', True)
                self.setBool('display_screen', True)

        self.random_seed = random_seed
        self.frame_skip = frame_skip
        self.state_frames = state_frames
        self.minimum_actions = minimum_actions
        self.image_processing = image_processing
        self.num_actions = 0
        self.legal_actions = []
        self.queue = deque()
        self.height = -1
        self.width = -1

    def loadROM(self, rom_file):
        ALEInterface.loadROM(self, rom_file)
        if self.minimum_actions:
            self.legal_actions = self.getMinimalActionSet()
        else:
            self.legal_actions = self.getLegalActionSet()
        self.num_actions = len(self.legal_actions)
        self.setInt('frame_skip', self.frame_skip)
        if self.random_seed is not None:
            self.setInt('random_seed', self.random_seed)
        self.height, self.width = self.getScreenDims()

    def act(self, a):
        return ALEInterface.act(self, self.legal_actions[a])

    def press_fire(self):
        return ALEInterface.act(self, 1)

    def saveScreenGrayscale(self, filename):
        im = self.getScreenGrayscale().squeeze(2)
        plt.close()
        plt.gray()
        plt.imsave(filename, im)

    def saveProcessedImage(self, filename):
        im = self.getProcessedImage()
        plt.close()
        plt.gray()
        plt.imsave(filename, im)

    def getScreenDims(self):
        w, h = ALEInterface.getScreenDims(self)
        return h, w  #dp

    def getProcessedImage(self):
        img = self.getScreenGrayscale()
        if self.image_processing is not None:
            return self.image_processing(img)
        else:
            return img
