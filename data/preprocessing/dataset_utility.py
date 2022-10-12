import csv
import glob
import os

import pandas as pd
import numpy as np
import tqdm
import mediapipe as mp
from utility.functions import load_pickle

class DatasetInformation:
    def __init__(self, name, videos, class_files, frames_root,
                 kinect_root, openpose_root,
                 dataset_mean, dataset_std, class_attribute_file, checkpoint_directory='', mediapipe_root=''):
        self.name = name
        self.videos = videos
        self.class_files = class_files
        self.frames_root = frames_root
        self.kinect_root = kinect_root
        self.openpose_root = openpose_root
        self.mediapipe_root = mediapipe_root
        self.checkpoint_directory = checkpoint_directory
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.class_attribute_file = class_attribute_file
        self.samples = []
        self.vocabulary = {}
        self.gesture_properties = {}


class Sample:
    def __init__(self, sign_id, class_id, description, signer_id, gesture_properties, phase, dataset, image_dims,
                 frames_path, kinect_path, openpose_path, mediapipe_path):
        self.sign_id = sign_id
        self.class_id = class_id
        self.description = description
        self.signer_id = signer_id
        self.phase = phase
        self.gesture_properties = gesture_properties
        self.dataset = dataset
        self.frames_path = frames_path
        self.openpose_path = openpose_path
        self.mediapipe_path = mediapipe_path
        self.kinect_path = kinect_path
        self.image_dims = image_dims


def split_datasets(args):
    datasets = list(args.datasets.keys())
    for ds in datasets:
        if ds == 'bsign22k':
            with open(args.datasets['bsign22k'].class_files) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                class_index = -1
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        print('Column names are :')
                        print([x for x in row])
                        line_count += 1
                    else:
                        if int(class_index) < args.max_num_classes or args.max_num_classes == -1:
                            args.datasets['bsign22k'].vocabulary[class_index] = 'bsign22k_' + row[1] + '_' + row[2]
                            args.datasets['bsign22k'].gesture_properties[class_index] = {'ClassID': row[1],
                                                                                         'ClassNameTr': row[2],
                                                                                         'One Hand': int(row[4]),
                                                                                         'Two Hand': int(row[6]),
                                                                                         'Circular': int(row[7]),
                                                                                         'Repetitive': int(row[8]),
                                                                                         'Mono': int(row[9]),
                                                                                         'Compound': int(row[10]),
                                                                                         'Chalearn Corresponding': row[
                                                                                             12]}

                    class_index += 1
                    line_count += 1
                print(f'Processed {line_count} lines.')
                AUTSL_mapping = {}
                for i in range(0, len(args.datasets['bsign22k'].gesture_properties)):
                    if args.datasets['bsign22k'].gesture_properties[i]['Chalearn Corresponding'] != '':
                        AUTSL_mapping[i] = args.datasets['bsign22k'].gesture_properties[i]['Chalearn Corresponding']
                AUTSL_mapping = {int(v): k for k, v in AUTSL_mapping.items()}
            video_list = sorted(glob.glob(args.datasets['bsign22k'].videos + os.sep + '*' + os.sep + '*.mp4'))
            for v in video_list:
                class_id = os.path.split(os.path.split(v)[0])[1]

                args.datasets['bsign22k'].samples.append(
                    Sample(sign_id=class_id + os.sep + os.path.splitext(os.path.split(v)[1])[0],
                           class_id=int(class_id) - 1,
                           description=args.datasets['bsign22k'].vocabulary[int(class_id) - 1],
                           signer_id=os.path.split(v)[1].split('_')[1],
                           phase='train' if (int(os.path.split(v)[1].split('_')[1]) not in args.bsign_val_user_index) else 'val',
                           dataset='bsign22k',
                           image_dims=[1920, 1080],
                           gesture_properties=args.datasets['bsign22k'].gesture_properties[int(class_id) - 1],
                           frames_path=os.path.join(args.datasets['bsign22k'].frames_root, class_id,os.path.splitext(os.path.split(v)[1])[0]),
                           openpose_path=os.path.join(args.datasets['bsign22k'].openpose_root, class_id,os.path.splitext(os.path.split(v)[1])[0]),
                           kinect_path=os.path.join(args.datasets['bsign22k'].kinect_root, class_id,os.path.splitext(os.path.split(v)[1])[0]),
                           mediapipe_path=os.path.join(args.datasets['bsign22k'].mediapipe_root, class_id,os.path.splitext(os.path.split(v)[1])[0])
                           ))


            print('Done processing Bosphorus sign config.')
        if ds == 'AUTSL':
            AUTSL_mapping = {1: 4, 8: 12, 9: 14, 11: 15, 14: 30, 15: 35, 20: 49, 22: 561, 28: 70, 29: 72, 35: 108,
                             42: 122,
                             52: 150, 58: 656, 61: 175, 62: 375, 64: 188, 66: 190, 73: 204, 76: 213, 77: 214, 82: 230,
                             83: 233, 94: 245, 96: 228, 100: 272, 101: 278, 103: 284, 112: 369, 116: 279, 117: 323,
                             122: 329, 125: 346, 128: 339, 141: 373, 144: 382, 151: 401, 154: 197, 163: 410, 165: 422,
                             172: 434, 176: 436, 177: 1, 183: 464, 188: 484, 193: 490, 200: 498, 204: 504, 205: 518,
                             206: 527, 209: 586, 211: 534, 215: 540, 217: 545, 221: 552, 222: 553, 224: 557}

            with open(args.datasets['AUTSL'].class_attribute_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                class_index = -1
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        print('Column names are :')
                        print([x for x in row])
                        line_count += 1
                    else:
                        if int(class_index) < args.max_num_classes or args.max_num_classes == -1:
                            args.datasets['AUTSL'].vocabulary[class_index] = 'AUTSL_' + row[1] + '_' + row[2]
                            args.datasets['AUTSL'].gesture_properties[class_index] = {'ClassID': row[0],
                                                                                      'ClassNameTr': row[1],
                                                                                      'ClassNameEng': row[2],
                                                                                      'One Hand': int(row[3]),
                                                                                      'Two Hand': int(row[6]),
                                                                                      'Circular': int(row[7]),
                                                                                      'Repetitive': int(row[4]),
                                                                                      'Mono': int(row[8]),
                                                                                      'Compound': int(row[9])}

                    class_index += 1
                    line_count += 1
                print(f'Processed {line_count} lines.')

            for phase in args.datasets['AUTSL'].class_files:
                phase_key = 'train' if ('train_labels' in phase) else 'val'
                phase_key_path = os.path.split(phase)[1].replace('_labels.csv', '')

                train_split_csv = pd.read_csv(phase, delimiter=',', header=None, dtype=str)
                for sample_idx, (video_id, class_id) in train_split_csv.iterrows():
                    if (int(class_id) < args.max_num_classes or args.max_num_classes == -1) and video_id not in [
                        'signer29_sample536', 'signer29_sample562', 'signer36_sample347']:  # broken video
                        signer_id = int(video_id.split('_')[0][6:])



                        sample = Sample(sign_id=phase_key + os.sep + video_id,
                               class_id=int(class_id),
                               description='AUTSL' + class_id,
                               signer_id=signer_id,
                               phase=phase_key,
                               dataset='AUTSL',
                               image_dims=[500, 500],
                               gesture_properties=args.datasets['AUTSL'].gesture_properties[int(class_id)],
                               frames_path=os.path.join(args.datasets['AUTSL'].frames_root,
                                                        phase_key_path + os.sep + video_id + '_color'),
                               openpose_path=os.path.join(args.datasets['AUTSL'].openpose_root,
                                                          phase_key_path + os.sep + video_id),
                               kinect_path=os.path.join(args.datasets['AUTSL'].kinect_root,
                                                        phase_key_path + os.sep + video_id + '_color'),
                               mediapipe_path = os.path.join(args.datasets['AUTSL'].mediapipe_root,
                                                   phase_key_path + os.sep + video_id + '_color'))
                        if int(class_id) in AUTSL_mapping.keys():
                            """
                            if args.datasets['AUTSL'].gesture_properties[int(class_id)]['Compound'] != [
                                args.datasets['bsign22k'].gesture_properties[AUTSL_mapping[int(class_id)]] if int(
                                    class_id) in AUTSL_mapping.keys() else {'Chalearn Corresponding': ''}][0]['Compound']:
                                print('here')
                            """
                            sample.gesture_properties['Chalearn Corresponding'] = int(class_id)
                        args.datasets['AUTSL'].samples.append(sample)

            for x in args.datasets['AUTSL'].samples:
                args.datasets['AUTSL'].vocabulary[x.class_id] = x.description
            print('Done processing AUTSL config.')
        if ds == 'msasl':
            # TODO msasl
            assert False, 'msasl config missing'

    args = calculate_joint_class_indices(args)


    return args

def calculate_joint_class_indices(args):


    args.combined_vocabulary_class_id = {}
    shared_gestures = []
    cnt = 0

    if args.using_autsl_gestures:
        for cl in args.datasets['AUTSL'].vocabulary.keys():
            args.combined_vocabulary_class_id[args.datasets['AUTSL'].vocabulary[cl]] = cl
            cnt += 1
    for cl in args.datasets['bsign22k'].vocabulary.keys():
        if args.datasets['bsign22k'].gesture_properties[cl]['Chalearn Corresponding'] == '':
            args.combined_vocabulary_class_id[args.datasets['bsign22k'].vocabulary[cl]] = cnt
            cnt +=1
        else:
            args.combined_vocabulary_class_id[args.datasets['bsign22k'].vocabulary[cl]] = int(args.datasets['bsign22k'].gesture_properties[cl]['Chalearn Corresponding'])
            shared_gestures.append(args.datasets['bsign22k'].vocabulary[cl])
            shared_gestures.append(args.datasets['AUTSL'].vocabulary[int(args.datasets['bsign22k'].gesture_properties[cl]['Chalearn Corresponding'])])

    if not args.using_nonshared_gestures:
        sign_keys = list(args.combined_vocabulary_class_id.keys())
        for sign_key in sign_keys:
            if sign_key not in shared_gestures:
                del args.combined_vocabulary_class_id[sign_key]
    args.combined_vocabulary_mapping = {x:i for i, x in enumerate(np.unique([x for x in args.combined_vocabulary_class_id.values()]))}


    return args


def preload_openpose_coordinates(args):
    coords = {}
    for ds in args.datasets.keys():
        ds_coords = {}
        for sm in tqdm.tqdm(args.datasets[ds].samples):
            if ds != 'AUTSL':
                openpose = load_pickle(sm.openpose_path)
            else:
                openpose = load_pickle(sm.openpose_path + '_color')
            ds_coords[sm.sign_id] = openpose
        coords[ds] = ds_coords
    return coords

def preload_mediapipe_coordinates(args):
    coords = {}
    for ds in args.datasets.keys():
        ds_coords = {}
        for sm in tqdm.tqdm(args.datasets[ds].samples):
            if ds != 'AUTSL':
                mediapipe = load_pickle(sm.mediapipe_path)
            else:
                mediapipe = load_pickle(sm.mediapipe_path.replace('_color',''))
            ds_coords[sm.sign_id] = mediapipe
        coords[ds] = ds_coords
    return coords


def detect_active_frames(args, coordinates):
    for ds in args.datasets.keys():
        print('\n' + ds + ' active frame extraction')
        for sm in tqdm.tqdm(args.datasets[ds].samples):
            if args.coordinate_detection_library == 'openpose':
                openpose = coordinates[ds][sm.sign_id]
                threshold_mat = ((((openpose['pose']['left_hip'][:, 1] + openpose['pose']['right_hip'][:, 1]) / 2) * 7) +
                                 openpose['pose']['neck'][:, 1]) / 8
                active_frames = np.minimum(openpose['hand_left']['lunate_bone'][:, 1],
                                           openpose['hand_right']['lunate_bone'][:, 1]) < threshold_mat
            elif args.coordinate_detection_library == 'mediapipe':
                mediapipe = coordinates[ds][sm.sign_id]['keypoints']
                hip_center = (mediapipe['pose'][mp.solutions.holistic.PoseLandmark.LEFT_HIP][:, 1] +
                                    mediapipe['pose'][mp.solutions.holistic.PoseLandmark.RIGHT_HIP][:, 1]) / 2
                shoulder_center = (mediapipe['pose'][mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER][:, 1] +
                                   mediapipe['pose'][mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER][:, 1])/2
                threshold_mat = ((hip_center * 7) + shoulder_center) / 8
                active_frames = np.minimum(mediapipe['hand_left'][mp.solutions.holistic.HandLandmark.WRIST][:, 1],
                                           mediapipe['hand_right'][mp.solutions.holistic.HandLandmark.WRIST][:, 1]) < threshold_mat
            else:
                assert False, 'Unknown Corrdinates'
            # sanity_check 1
            dont_use_active_frames = False
            if np.any(active_frames):
                first_up = np.where(active_frames)[0][0]
                last_up = np.where(active_frames)[0][-1]
                if last_up - first_up < 10:
                    dont_use_active_frames = True
            else:
                dont_use_active_frames = True

            # sanity_check
            if dont_use_active_frames:
                first_up = 0
                last_up = len(active_frames) - 1
            sm.active_frame_range = [first_up, last_up]

    return args
