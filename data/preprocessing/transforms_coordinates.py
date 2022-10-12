from data.preprocessing import transforms_functional as F

import numbers
import random
import numpy as np
import PIL
import torchvision
import torch


class RandomChoose(object):

    def __init__(self, p=0.5, keep_ratio=0.8):
        self.p = p
        self.keep_ratio = keep_ratio

    def __call__(self, coords):
        if random.random() < self.p:
            if isinstance(coords[0], np.ndarray):
                return [np.fliplr(img) for img in coords]

            else:
                raise TypeError('Expected numpy.ndarray but got list of {0}'.format(type(coords[0])))


        return coords


class RandomShift(object):

    def __init__(self, p=0.5, keep_ratio=0.8):
        self.p = p
        self.keep_ratio = keep_ratio

    def __call__(self, coords):
        if random.random() < self.p:
            if isinstance(coords[0], np.ndarray):
                coords[:, 0, :] += random.random() * 20 - 10.0
                coords[:, 1, :] += random.random() * 20 - 10.0
            else:
                raise TypeError('Expected numpy.ndarray but got {0}'.format(type(coords)))

        return coords


class RandomMirror(object):

    def __init__(self, p=0.5, image_size=256):
        self.p = p
        self.image_size = image_size

    def __call__(self, coords):
        if random.random() < self.p:
            if isinstance(coords[0], np.ndarray):
                coords[:, 0, :] = self.image_size - coords[:, 0, :]
            else:
                raise TypeError('Expected numpy.ndarray but got {0}'.format(type(coords)))


        return coords


class Normalization(object):

    def __init__(self):
        pass
    def __call__(self, coords):
        if isinstance(coords[0], np.ndarray):

            coords[:, 0, :] = coords[:, 0, :] - coords[0, 0, np.sum(np.sum(coords,axis=0),axis=0)>0].mean()
            coords[:, 1, :] = coords[:, 1, :] - coords[0, 1, np.sum(np.sum(coords,axis=0),axis=0)>0].mean()
        else:
            raise TypeError('Expected numpy.ndarray but got list of {0}'.format(type(coords[0])))
        return coords