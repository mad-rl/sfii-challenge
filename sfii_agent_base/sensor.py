import numpy as np
from PIL import Image


class Sensor():
    def __init__(self, width=84, height=84):
        self.width = width
        self.height = height

    def preprocess_obs(self, observation):
        image = Image.fromarray(observation).convert('L')
        image = image.resize((self.width, self.height))
        image_array = np.array(image).astype('float32')
        image_array = image_array / 255

        return image_array
