from torch.utils import data
from matplotlib import pyplot as plt
from typing import List
from data.joint_groups import choose_joint_groups

import itertools
import PIL
import time
import numpy as np
import glob
import os


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(self.interval)


class SignLanguageDataset(data.Dataset):
    _feature_ch_info = {
        'full': 448,
        'hand_right': 128,
        'hand_left': 128,
        'face': 128}

    def __init__(self,
                 args=None,
                 preloaded_data=None,
                 split: str = None,
                 transform=None):
        self.args = args
        self.preloaded_data = preloaded_data
        self.split = split
        self.vocabulary = [self.args.datasets[ds].vocabulary for ds in self.args.datasets.keys()]
        kj, oj, gj = choose_joint_groups(args)
        self.openpose_joint_groups = oj

        self.samples = list(
            itertools.chain.from_iterable([self.args.datasets[ds].samples for ds in self.args.datasets.keys()]))
        # Select Only Samples Belonging to set
        self.samples = [x for x in self.samples if x.phase == split]

        self.transform = transform
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        coords, joint_names = self.load_coordinates(item)
        keyframes, active_frame_range, T, T_active = self.load_keyframes(item, coords)
        label = self.samples[item].class_id

        # Pad Coordinates
        padded_coords = np.zeros((coords.shape[0], self.args.clip_length, coords.shape[2]), dtype=np.float16)
        if T_active > self.args.clip_length:
            padding_map = np.linspace(0, T_active-1, self.args.clip_length, dtype=int)
            padded_coords = coords[:, padding_map, :]
            # update keyframes according to linear interpolation & preserve empty keyframes if less than maximum detected
            preserve_empty = keyframes[keyframes == -1]
            reverse_padding_map = self.extract_reverse_padding_map(T_active-1, padding_map)
            keyframes = reverse_padding_map[keyframes]
            keyframes[preserve_empty] = -1
        elif T_active < self.args.clip_length:
            if self.args.temporal_sampling == 'linear_sampling':
                padding_map = np.zeros(self.args.clip_length, dtype=int)
                padding_map[0:T_active] = np.linspace(0, T_active-1, T_active, dtype=int)
                padded_coords[:, 0:T_active, :] = coords
                assert False, 'Not complete'
            elif self.args.temporal_sampling == 'pad_zeros':
                padding_map = np.zeros(self.args.clip_length, dtype=int)
                padding_map[0:T_active] = np.linspace(0, T_active-1, T_active, dtype=int)
                padded_coords[:, 0:T_active, :] = coords
            else:
                assert False, 'Not Implemented'

        else:  # T_active == self.args.clip_length:
            padding_map = np.linspace(0, T_active-1, self.args.clip_length, dtype=int)
            padded_coords = coords


        name = self.samples[item].description
        sign_id = self.samples[item].sign_id

        padded_coords = padded_coords.transpose([0,2,1])
        # Correct Upside down
        #padded_coords[:, 1, :] = self.args.input_size - padded_coords[:, 1, :]
        sample = {'coords': padded_coords, 'padding_map': padding_map, 'keyframes': keyframes, }
        sample = self.transform(sample)
        padded_coords = sample['coords']
        padding_map = sample['padding_map']
        keyframes = sample['keyframes']

        return padded_coords, label, keyframes, padding_map, active_frame_range, T, name, sign_id, joint_names
        # return features, label, name, signer, seq_length, coords, []

    def extract_reverse_padding_map(self, T_active, padding_map):
        reverse_padding_map = np.zeros((T_active), dtype=int)
        for x in range(T_active):
            if x in padding_map:
                ind = np.where(padding_map == x)[0][0]
                reverse_padding_map[x] = ind
                reverse_padding_map[reverse_padding_map == -1] = ind
            else:
                reverse_padding_map[x] = -1
        reverse_padding_map[reverse_padding_map == -1] = padding_map.shape[0]-1
        return reverse_padding_map

    def load_coordinates(self, item):
        openpose_coords = self.preloaded_data['openpose'][self.samples[item].dataset][self.samples[item].sign_id]
        coords = []
        joint_names = []
        for jnt in self.openpose_joint_groups:
            coords.append(openpose_coords[jnt[0]][jnt[1]])
            joint_names.append(jnt[0] + '_' + jnt[1])
        coords = np.array(coords)[:,
                 self.samples[item].active_frame_range[0]:self.samples[item].active_frame_range[1] + 1, 0:3]
        coords[:, :, 0] = coords[:, :, 0] / self.samples[item].image_dims[0] * self.args.input_size
        coords[:, :, 1] = coords[:, :, 1] / self.samples[item].image_dims[1] * self.args.input_size
        coords[:, :, 2] = coords[:, :, 2]

        return coords, joint_names

    def load_keyframes(self, item, coords):
        T = coords.shape[1]
        T_active = self.samples[item].active_frame_range[1] - self.samples[item].active_frame_range[0] + 1
        keyframes = np.array(self.samples[item].keyframes) - self.samples[item].active_frame_range[0]
        keyframes = np.sort(keyframes)
        # If number of keyframes is smaller than fixed size
        while keyframes.shape[0] < self.args.no_of_static_keyframes:
            keyframes = np.append(keyframes, -1)
        return keyframes, self.samples[item].active_frame_range, T, T_active