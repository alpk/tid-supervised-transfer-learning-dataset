from torch.utils import data
from matplotlib import pyplot as plt

from data.joint_groups import choose_joint_groups

import csv
import itertools
import PIL
import time
import numpy as np
import glob
import os
import torch
import json
import random

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(self.interval)


ATTR_NAMES = ["One Hand", "Two Hand", "Circular", "Repetitive", "Mono", "Compound"]

class SignLanguageDataset(data.Dataset):
    def __init__(self,
                 args=None,
                 split: str = None,
                 transform=None,
                 transfer_split='',
                 preloaded_data=None):
        self.args = args
        self.split = split
        self.preloaded_data = preloaded_data
        self.vocabulary = [self.args.datasets[ds].vocabulary for ds in self.args.datasets.keys()]
        kj, oj, gj = choose_joint_groups(args)
        self.coordinate_joint_groups = oj

        input_file = csv.DictReader(open('./configuration/json/' + transfer_split+ '.csv'))

        samples_dict = [row for row in input_file] 
        self.signer_id = set(list(map(lambda x:x["signer_id"], samples_dict)))
        self.signer_id = sorted(self.signer_id)
        # print(self.signer_id)
        if split != "val":
            if "bsign22k" in transfer_split and args.bsign_user_index[0] != "all":
                self.signer_id = list(map(str, args.bsign_user_index))
            elif 'AUTSL' in transfer_split and args.autsl_user_index[0] != "all":
                self.signer_id = list(map(str, args.autsl_user_index))

        self.samples = list(filter(lambda x: x["signer_id"] in self.signer_id, samples_dict))
        self.unique_labels = np.unique([(int(x['class_id'])) for x in self.samples])
        if args.using_nonshared_gestures:
            if 'whole' in transfer_split:
                if 'AUTSL' in transfer_split:
                    self.label_mappings = {x: x for i, x in enumerate(self.unique_labels)}
                else:
                    #assert False, 'Check here to implement bsign whole'
                    self.label_mappings = {x: i for i, x in enumerate(self.unique_labels)}
            else:
                self.label_mappings = {x: x for i, x in enumerate(self.unique_labels)}
        else:
            self.samples = list(filter(lambda x: x["signer_id"] in self.signer_id, samples_dict))
            self.unique_labels = np.unique([(int(x['class_id'])) for x in self.samples])
            self.label_mappings = {x: i for i, x in enumerate(self.unique_labels)}
        #TODO check if all works well
        if self.args.use_multilabel != 'no':
            for x in self.samples:
                x['multilabel_features'] = preloaded_data['signwriting'][x['dataset']][x['sign_id']]
        print(self.label_mappings)
        self.transform = transform

        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        #coords, joint_names = self.load_coordinates(item)
        active_frame_range = [int(self.samples[item]['active_frame_range_start']),
                              int(self.samples[item]['active_frame_range_end'])]
        T_active = active_frame_range[1] - active_frame_range[0]

        features, joint_names = self.load_inputs(item, active_frame_range, T_active)

        label = self.label_mappings[int(self.samples[item]['class_id'])]
        unmapped_label = int(self.samples[item]['class_id'])
        name = self.samples[item]['description']
        sample_id = self.samples[item]['sign_id']
        signer_id = int(self.samples[item]['signer_id'])


        attributes = list(map(lambda x:self.samples[0][x], ATTR_NAMES))

        features = self.transform(features)
        multilabel_features = int(self.samples[item]['signer_id'])
        if self.args.use_multilabel != 'no':
            multilabel_features = self.samples[item]['multilabel_features']


        return features, label , name, signer_id, sample_id, multilabel_features
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


    def load_active_frames(self,item, coords):
        T = coords.shape[1]
        T_active = self.samples[item].active_frame_range[1] - self.samples[item].active_frame_range[0] + 1
        return self.samples[item].active_frame_range, T, T_active

    def load_keyframes(self, item, coords):
        T = coords.shape[1]
        T_active = self.samples[item].active_frame_range[1] - self.samples[item].active_frame_range[0] + 1
        keyframes = np.array(self.samples[item].keyframes) - self.samples[item].active_frame_range[0]
        keyframes = np.sort(keyframes)
        # If number of keyframes is smaller than fixed size
        while keyframes.shape[0] < self.args.no_of_static_keyframes:
            keyframes = np.append(keyframes, -1)
        return keyframes, self.samples[item].active_frame_range, T, T_active

    def load_inputs(self, item, active_frame_range, T_active):
        if self.args.model_type in ['GCN']:
            inputs, joint_names = self.load_coordinates(item, active_frame_range, T_active)
        else:
            inputs = self.load_frames(item, active_frame_range, T_active)
            joint_names = None
        return inputs , joint_names

    def load_coordinates(self, item, active_frame_range, T_active):
        openpose_coords = self.preloaded_data[self.args.coordinate_detection_library][self.samples[item]['dataset']][self.samples[item]['sign_id']]
        coords = []
        joint_names = []
        if self.args.coordinate_detection_library == 'mediapipe':
            for jnt in self.coordinate_joint_groups:
                buffer = np.concatenate([openpose_coords['keypoints'][jnt[0]][jnt[1]][:, 0:2],
                                openpose_coords['confidences'][jnt[0]][jnt[1]]], axis=1)
                coords.append(buffer)
                joint_names.append(jnt[0] + '_' + str(jnt[1]))
        else:
            for jnt in self.coordinate_joint_groups:
                coords.append(openpose_coords[jnt[0]][jnt[1]] )
                joint_names.append(jnt[0] + '_' + str(jnt[1]))

        coords = np.array(coords)[:, active_frame_range[0]:active_frame_range[1]+1, 0:3]

        frame_list = list(range(active_frame_range[0],active_frame_range[1]+1))
        T = len (frame_list)
        frame_indices = self.get_segment_indices(frame_list, T)
        coords = coords[:,frame_indices,:]
        x_scale = json.loads(self.samples[item]['image_dims'])[0] if self.args.coordinate_detection_library == 'openpose' else 1.0
        y_scale = json.loads(self.samples[item]['image_dims'])[1] if self.args.coordinate_detection_library == 'openpose' else 1.0
        coords[:, :, 0] = coords[:, :, 0] / x_scale * self.args.input_size
        coords[:, :, 1] = coords[:, :, 1] / y_scale * self.args.input_size
        coords[:, :, 2] = coords[:, :, 2]

        coords = np.transpose(coords,[0,2,1])

        return coords, joint_names



    def load_frames(self, item, active_frame_range, T_active):
        frame_path = self.samples[item]['frames_path'] + os.sep
        frame_list = glob.glob(os.path.join(frame_path, '*.png'))
        frame_list.extend(glob.glob(os.path.join(frame_path, '*.jpg')))
        frame_list = sorted(frame_list)
        T = len(frame_list)
        if T_active > 10:
            frame_list = np.array(frame_list)[np.arange(active_frame_range[0], active_frame_range[1])-1]
        else:
            frame_list = np.array(frame_list)
        # TODO kontrol et
        frame_indices = self.get_segment_indices(frame_list, T)

        frame_list = np.array(frame_list)
        input = []
        for idx, frame in enumerate(frame_list[frame_indices]):
            image = PIL.Image.open(str(frame))
            input.append(image)

        # return input, frame_indices
        return input


    def get_segment_indices(self, frame_list, T):
        frame_indices = []
        if self.args.temporal_sampling == 'linear_sampling':
            frame_indices = np.linspace(0, len(frame_list) - 1, self.args.clip_length, dtype=np.int)
        elif self.args.temporal_sampling == 'random_linear_sampling':
            step_size = max(1, (len(frame_list) - 1) // self.args.clip_length // 2)
            random_step = np.random.randint(-step_size, step_size, self.args.clip_length)
            frame_indices = np.linspace(0, len(frame_list) - 1, self.args.clip_length, dtype=np.int) + random_step
            frame_indices[0] = max(frame_indices[0], 0)
            frame_indices[-1] = min(frame_indices[-1], len(frame_list) - 1)
        elif self.args.temporal_sampling == 'pad_zeros':
            frame_indices = np.linspace(0, len(frame_list) - 1, self.args.clip_length, dtype=np.int)
        return frame_indices