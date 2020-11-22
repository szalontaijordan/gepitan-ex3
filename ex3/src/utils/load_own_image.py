import numpy as np
import cv2
from os import getcwd, path


def load_own_image(name):
    """
    Loads a custom 20x20 grayscale image to a (1, 400) vector.

    :param name: name of the image file
    :return: (1, 400) vector of the grayscale image
    """

    print('Loading image:', name)

    file_name = path.join(getcwd(), 'ex3', 'src', 'data', name)
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    # reshape 20x20 grayscale image to a vector
    return np.reshape(img.T / 255, (1, 400))
