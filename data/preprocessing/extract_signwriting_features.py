import mediapipe.python.solutions.holistic
import numpy as np
import tqdm
import os
import pickle
import mediapipe as mp
mp_holistic = mp.solutions.holistic
from matplotlib import pyplot as plt


normalization_constant = 1080


def movement_feature(sample, coords, signwriting_features_list, args):
    feat = []
    def add_distance(coord1, index_finger_tip, thumb_finger_tip):
        return np.min([
            np.min(np.linalg.norm(coord1[:, 0:2] - index_finger_tip[:, 0:2], axis=1)),
            np.min(np.linalg.norm(coord1[:, 0:2] - thumb_finger_tip[:, 0:2], axis=1))])

    if args.coordinate_detection_library == 'openpose':
        left_index_finger_tip = coords['hand_left']['index_finger_8']
        left_thumb_finger_tip = coords['hand_left']['thumb_4']
        right_index_finger_tip = coords['hand_right']['index_finger_8']
        right_thumb_finger_tip = coords['hand_right']['thumb_4']


        if 'lh_touch_upper_face' in signwriting_features_list:
            feat.append(add_distance((coords['face']['left_eyebrow_22'] + coords['face']['right_eyebrow_21']) / 2, left_index_finger_tip,left_thumb_finger_tip))
        if 'rh_touch_upper_face' in signwriting_features_list:
            feat.append(add_distance((coords['face']['left_eyebrow_22'] + coords['face']['right_eyebrow_21']) / 2, right_index_finger_tip,right_thumb_finger_tip))
        if 'lh_touch_nose' in signwriting_features_list:
            feat.append(add_distance(coords['face']['nose_33'], left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_nose' in signwriting_features_list:
            feat.append(add_distance(coords['face']['nose_33'], right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_left_ear' in signwriting_features_list:
            feat.append(add_distance(coords['face']['face_border_0'], left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_left_ear' in signwriting_features_list:
            feat.append(add_distance(coords['face']['face_border_0'], right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_right_ear' in signwriting_features_list:
            feat.append(add_distance(coords['face']['face_border_16'], left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_right_ear' in signwriting_features_list:
            feat.append(add_distance(coords['face']['face_border_16'], right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_left_eye' in signwriting_features_list:
            feat.append(add_distance(coords['pose']['left_eye'], left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_left_eye' in signwriting_features_list:
            feat.append(add_distance(coords['pose']['left_eye'], right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_right_eye' in signwriting_features_list:
            feat.append(add_distance(coords['pose']['right_eye'], left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_right_eye' in signwriting_features_list:
            feat.append(add_distance(coords['pose']['right_eye'], right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_left_cheek' in signwriting_features_list:
            feat.append(add_distance(coords['face']['face_border_5'], left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_left_cheek' in signwriting_features_list:
            feat.append(add_distance(coords['face']['face_border_5'], right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_right_cheek' in signwriting_features_list:
            feat.append(add_distance(coords['face']['face_border_11'], left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_right_cheek' in signwriting_features_list:
            feat.append(add_distance(coords['face']['face_border_11'], right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_mouth' in signwriting_features_list:
            feat.append(add_distance(coords['face']['mouth_66'], left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_mouth' in signwriting_features_list:
            feat.append(add_distance(coords['face']['mouth_66'], right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_chin' in signwriting_features_list:
            feat.append(add_distance(coords['face']['face_border_8'], left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_chin' in signwriting_features_list:
            feat.append(add_distance(coords['face']['face_border_8'], right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_neck' in signwriting_features_list:
            feat.append(add_distance(coords['pose']['neck'], left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_neck' in signwriting_features_list:
            feat.append(add_distance(coords['pose']['neck'], right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_left_shoulder' in signwriting_features_list:
            feat.append(add_distance(coords['pose']['left_shoulder'], left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_left_shoulder' in signwriting_features_list:
            feat.append(add_distance(coords['pose']['left_shoulder'], right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_right_shoulder' in signwriting_features_list:
            feat.append(add_distance(coords['pose']['right_shoulder'], left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_right_shoulder' in signwriting_features_list:
            feat.append(add_distance(coords['pose']['right_shoulder'], right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_chest' in signwriting_features_list:
            feat.append(add_distance((coords['pose']['right_shoulder'] + coords['pose']['left_shoulder'])/2, left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_chest' in signwriting_features_list:
            feat.append(add_distance((coords['pose']['right_shoulder'] + coords['pose']['left_shoulder'])/2, right_index_finger_tip, right_thumb_finger_tip))
    else:
        left_index_finger_tip = coords['keypoints']['hand_left'][mp_holistic.HandLandmark.INDEX_FINGER_TIP]
        left_thumb_finger_tip = coords['keypoints']['hand_left'][mp_holistic.HandLandmark.THUMB_TIP]
        right_index_finger_tip = coords['keypoints']['hand_right'][mp_holistic.HandLandmark.INDEX_FINGER_TIP]
        right_thumb_finger_tip = coords['keypoints']['hand_right'][mp_holistic.HandLandmark.THUMB_TIP]

        if 'lh_touch_upper_face' in signwriting_features_list:
            feat.append(add_distance((coords['keypoints']['pose'][mp_holistic.PoseLandmark.LEFT_EYE] +
                                      coords['keypoints']['pose'][mp_holistic.PoseLandmark.RIGHT_EYE]) / 2,
                                     left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_upper_face' in signwriting_features_list:
            feat.append(add_distance((coords['keypoints']['pose'][mp_holistic.PoseLandmark.LEFT_EYE] +
                                      coords['keypoints']['pose'][mp_holistic.PoseLandmark.RIGHT_EYE]) / 2,
                                     right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_nose' in signwriting_features_list:
            feat.append(add_distance(coords['keypoints']['pose'][mp_holistic.PoseLandmark.NOSE], left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_nose' in signwriting_features_list:
            feat.append(add_distance(coords['keypoints']['pose'][mp_holistic.PoseLandmark.NOSE], right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_left_ear' in signwriting_features_list:
            feat.append(add_distance(coords['keypoints']['pose'][mp_holistic.PoseLandmark.LEFT_EAR], left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_left_ear' in signwriting_features_list:
            feat.append(add_distance(coords['keypoints']['pose'][mp_holistic.PoseLandmark.LEFT_EAR], right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_right_ear' in signwriting_features_list:
            feat.append(add_distance(coords['keypoints']['pose'][mp_holistic.PoseLandmark.RIGHT_EAR], left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_right_ear' in signwriting_features_list:
            feat.append(add_distance(coords['keypoints']['pose'][mp_holistic.PoseLandmark.RIGHT_EAR], right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_left_eye' in signwriting_features_list:
            feat.append(add_distance(coords['keypoints']['pose'][mp_holistic.PoseLandmark.LEFT_EYE], left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_left_eye' in signwriting_features_list:
            feat.append(add_distance(coords['keypoints']['pose'][mp_holistic.PoseLandmark.LEFT_EYE], right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_right_eye' in signwriting_features_list:
            feat.append(add_distance(coords['keypoints']['pose'][mp_holistic.PoseLandmark.RIGHT_EYE], left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_right_eye' in signwriting_features_list:
            feat.append(add_distance(coords['keypoints']['pose'][mp_holistic.PoseLandmark.RIGHT_EYE], right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_left_cheek' in signwriting_features_list:
            feat.append(add_distance((coords['keypoints']['pose'][mp_holistic.PoseLandmark.MOUTH_LEFT] +
                                      coords['keypoints']['pose'][mp_holistic.PoseLandmark.LEFT_EAR])/2 , left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_left_cheek' in signwriting_features_list:
            feat.append(add_distance((coords['keypoints']['pose'][mp_holistic.PoseLandmark.MOUTH_LEFT] +
                                      coords['keypoints']['pose'][mp_holistic.PoseLandmark.LEFT_EAR])/2, right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_right_cheek' in signwriting_features_list:
            feat.append(add_distance((coords['keypoints']['pose'][mp_holistic.PoseLandmark.MOUTH_RIGHT] +
                                      coords['keypoints']['pose'][mp_holistic.PoseLandmark.RIGHT_EAR])/2, left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_right_cheek' in signwriting_features_list:
            feat.append(add_distance((coords['keypoints']['pose'][mp_holistic.PoseLandmark.MOUTH_RIGHT] +
                                      coords['keypoints']['pose'][mp_holistic.PoseLandmark.RIGHT_EAR]) / 2, right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_mouth' in signwriting_features_list:
            feat.append(add_distance((coords['keypoints']['pose'][mp_holistic.PoseLandmark.MOUTH_RIGHT] +
                                      coords['keypoints']['pose'][mp_holistic.PoseLandmark.MOUTH_LEFT]) / 2, left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_mouth' in signwriting_features_list:
            feat.append(add_distance((coords['keypoints']['pose'][mp_holistic.PoseLandmark.MOUTH_RIGHT] +
                                      coords['keypoints']['pose'][mp_holistic.PoseLandmark.MOUTH_LEFT]) / 2, right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_chin' in signwriting_features_list: # No Cheek Joint
            feat.append(add_distance((coords['keypoints']['pose'][mp_holistic.PoseLandmark.MOUTH_LEFT]+
                                      coords['keypoints']['pose'][mp_holistic.PoseLandmark.MOUTH_RIGHT] +
                                      coords['keypoints']['pose'][mp_holistic.PoseLandmark.LEFT_SHOULDER] +
                                      coords['keypoints']['pose'][mp_holistic.PoseLandmark.RIGHT_SHOULDER])/4,
                                     left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_chin' in signwriting_features_list:
            feat.append(add_distance((coords['keypoints']['pose'][mp_holistic.PoseLandmark.MOUTH_LEFT]+
                                      coords['keypoints']['pose'][mp_holistic.PoseLandmark.MOUTH_RIGHT] +
                                      coords['keypoints']['pose'][mp_holistic.PoseLandmark.LEFT_SHOULDER] +
                                      coords['keypoints']['pose'][mp_holistic.PoseLandmark.RIGHT_SHOULDER])/4,
                                     right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_neck' in signwriting_features_list:
            feat.append(add_distance((coords['keypoints']['pose'][mp_holistic.PoseLandmark.LEFT_SHOULDER] +
                                      coords['keypoints']['pose'][mp_holistic.PoseLandmark.RIGHT_SHOULDER]) / 2,
                                     left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_neck' in signwriting_features_list:
            feat.append(add_distance((coords['keypoints']['pose'][mp_holistic.PoseLandmark.LEFT_SHOULDER] +
                                      coords['keypoints']['pose'][mp_holistic.PoseLandmark.RIGHT_SHOULDER]) / 2,
                                     right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_left_shoulder' in signwriting_features_list:
            feat.append(add_distance(coords['keypoints']['pose'][mp_holistic.PoseLandmark.LEFT_SHOULDER],
                                     left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_left_shoulder' in signwriting_features_list:
            feat.append(add_distance(coords['keypoints']['pose'][mp_holistic.PoseLandmark.LEFT_SHOULDER],
                                     right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_right_shoulder' in signwriting_features_list:
            feat.append(add_distance(coords['keypoints']['pose'][mp_holistic.PoseLandmark.RIGHT_SHOULDER],
                                     left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_right_shoulder' in signwriting_features_list:
            feat.append(add_distance(coords['keypoints']['pose'][mp_holistic.PoseLandmark.RIGHT_SHOULDER],
                                     right_index_finger_tip, right_thumb_finger_tip))
        if 'lh_touch_chest' in signwriting_features_list:
            feat.append(add_distance((coords['keypoints']['pose'][mp_holistic.PoseLandmark.LEFT_SHOULDER] +
                                      coords['keypoints']['pose'][mp_holistic.PoseLandmark.RIGHT_SHOULDER]) / 2,
                                     left_index_finger_tip, left_thumb_finger_tip))
        if 'rh_touch_chest' in signwriting_features_list:
            feat.append(add_distance((coords['keypoints']['pose'][mp_holistic.PoseLandmark.LEFT_SHOULDER] +
                                      coords['keypoints']['pose'][mp_holistic.PoseLandmark.RIGHT_SHOULDER]) / 2,
                                     right_index_finger_tip, right_thumb_finger_tip))


    normalized_feat = np.array(feat) / normalization_constant
    return normalized_feat

def dominant_hand_movement(sample, coords, signwriting_features_list, args):
    feat = []
    if args.coordinate_detection_library == 'openpose':
        calc_min_distance = lambda coord1, coord2: np.min(np.linalg.norm(coord1[:, 0:2] - coord2[:, 0:2], axis=1))
        calc_distances = lambda coord1, coord2: np.linalg.norm(coord1[:, 0:2] - coord2[:, 0:2], axis=1)
        inter_eye_distance = calc_min_distance(coords['pose']['left_eye'], coords['pose']['right_eye'])
        if 'together' in signwriting_features_list:
            feat.append((np.min([
               calc_min_distance(coords['hand_left']['index_finger_8'], coords['hand_right']['index_finger_8']),
               calc_min_distance(coords['hand_left']['thumb_4'], coords['hand_right']['index_finger_8']),
               calc_min_distance(coords['hand_left']['index_finger_8'], coords['hand_right']['thumb_4']),
               calc_min_distance(coords['hand_left']['thumb_4'], coords['hand_right']['thumb_4'])]) < inter_eye_distance).astype(np.float))
        if 'apart' in signwriting_features_list:
            feat.append((np.min([
           calc_min_distance(coords['hand_left']['index_finger_8'], coords['hand_right']['index_finger_8']),
           calc_min_distance(coords['hand_left']['thumb_4'], coords['hand_right']['index_finger_8']),
           calc_min_distance(coords['hand_left']['index_finger_8'], coords['hand_right']['thumb_4']),
           calc_min_distance(coords['hand_left']['thumb_4'], coords['hand_right']['thumb_4'])]) / normalization_constant).astype(np.float))
        if 'Circular' in signwriting_features_list:
            feat.append(np.array(sample.gesture_properties['Circular'] if sample.dataset == 'bsign22k' else 0).astype(np.float))
        if 'Repetitive' in signwriting_features_list:
            feat.append(np.array(sample.gesture_properties['Repetitive'] if sample.dataset == 'bsign22k' else 0).astype(np.float))
        if 'up_down' in signwriting_features_list:
            feat.append(np.max((coords['hand_left']['index_finger_8'][:, 1]) - np.min(coords['hand_right']['index_finger_8'][:, 1])).astype(np.float) / normalization_constant)
        if 'side_to_side' in signwriting_features_list:
            feat.append(np.max((coords['hand_left']['index_finger_8'][:, 0]) - np.min(coords['hand_right']['index_finger_8'][:, 0])).astype(np.float) / normalization_constant)
        if 'Two Hand' in signwriting_features_list:
            feat.append( np.array(sample.gesture_properties['Two Hand'] if sample.dataset == 'bsign22k' else 0).astype(np.float))



        temp = np.abs(coords['hand_left']['index_finger_8'][sample.active_frame_range[0]:sample.active_frame_range[1], 1]
                    - coords['hand_right']['index_finger_8'][sample.active_frame_range[0]:sample.active_frame_range[1], 1])  < inter_eye_distance * 2
        cnt = 0
        cnt_max = 0
        for x in temp:
            if cnt:
                cnt += 1
            else:
                cnt_max = cnt
                cnt = 0

        if 'side_by_side' in signwriting_features_list:
            feat.append(np.array(cnt_max > 10).astype(np.float))
        temp = calc_distances(coords['hand_left']['index_finger_8'], coords['hand_right']['index_finger_8'])
        if 'contacting' in signwriting_features_list:
            feat.append(np.any(temp < inter_eye_distance).astype(np.float))
    else:
        calc_min_distance = lambda coord1, coord2: np.min(np.linalg.norm(coord1[:, 0:2] - coord2[:, 0:2], axis=1))
        calc_distances = lambda coord1, coord2: np.linalg.norm(coord1[:, 0:2] - coord2[:, 0:2], axis=1)
        inter_eye_distance = calc_min_distance(coords['keypoints']['pose'][mp_holistic.PoseLandmark.LEFT_EYE],
                                               coords['keypoints']['pose'][mp_holistic.PoseLandmark.RIGHT_EYE])
        if 'together' in signwriting_features_list:
            feat.append((np.min([
                calc_min_distance(coords['keypoints']['hand_left'][mp_holistic.HandLandmark.INDEX_FINGER_TIP], coords['keypoints']['hand_right'][mp_holistic.HandLandmark.INDEX_FINGER_TIP]),
                calc_min_distance(coords['keypoints']['hand_left'][mp_holistic.HandLandmark.THUMB_TIP],        coords['keypoints']['hand_right'][mp_holistic.HandLandmark.INDEX_FINGER_TIP]),
                calc_min_distance(coords['keypoints']['hand_left'][mp_holistic.HandLandmark.INDEX_FINGER_TIP], coords['keypoints']['hand_right'][mp_holistic.HandLandmark.THUMB_TIP]),
                calc_min_distance(coords['keypoints']['hand_left'][mp_holistic.HandLandmark.THUMB_TIP],
                                  coords['keypoints']['hand_right'][mp_holistic.HandLandmark.THUMB_TIP])]) < inter_eye_distance).astype(np.float))
        if 'apart' in signwriting_features_list:
            feat.append((np.min([
                calc_min_distance(coords['keypoints']['hand_left'][mp_holistic.HandLandmark.INDEX_FINGER_TIP], coords['keypoints']['hand_right'][mp_holistic.HandLandmark.INDEX_FINGER_TIP]),
                calc_min_distance(coords['keypoints']['hand_left'][mp_holistic.HandLandmark.THUMB_TIP],        coords['keypoints']['hand_right'][mp_holistic.HandLandmark.INDEX_FINGER_TIP]),
                calc_min_distance(coords['keypoints']['hand_left'][mp_holistic.HandLandmark.INDEX_FINGER_TIP], coords['keypoints']['hand_right'][mp_holistic.HandLandmark.THUMB_TIP]),
                calc_min_distance(coords['keypoints']['hand_left'][mp_holistic.HandLandmark.THUMB_TIP],
                                  coords['keypoints']['hand_right'][mp_holistic.HandLandmark.THUMB_TIP])]) / normalization_constant).astype(np.float))
        if 'Circular' in signwriting_features_list:
            feat.append(
                np.array(sample.gesture_properties['Circular'] if sample.dataset == 'bsign22k' else 0).astype(np.float))
        if 'Repetitive' in signwriting_features_list:
            feat.append(np.array(sample.gesture_properties['Repetitive'] if sample.dataset == 'bsign22k' else 0).astype(
                np.float))
        if 'up_down' in signwriting_features_list:
            feat.append(np.max((coords['keypoints']['hand_left'][mp_holistic.HandLandmark.INDEX_FINGER_TIP][:, 1]) - np.min(
                coords['keypoints']['hand_right'][mp_holistic.HandLandmark.INDEX_FINGER_TIP][:, 1])).astype(np.float) / normalization_constant)
        if 'side_to_side' in signwriting_features_list:
            feat.append(np.max((coords['keypoints']['hand_left'][mp_holistic.HandLandmark.INDEX_FINGER_TIP][:, 0]) - np.min(
                coords['keypoints']['hand_right'][mp_holistic.HandLandmark.INDEX_FINGER_TIP][:, 0])).astype(np.float) / normalization_constant)
        if 'Two Hand' in signwriting_features_list:
            feat.append(
                np.array(sample.gesture_properties['Two Hand'] if sample.dataset == 'bsign22k' else 0).astype(np.float))

        temp = np.abs(
            coords['keypoints']['hand_left'][mp_holistic.HandLandmark.INDEX_FINGER_TIP][sample.active_frame_range[0]:sample.active_frame_range[1], 1]
            - coords['keypoints']['hand_right'][mp_holistic.HandLandmark.INDEX_FINGER_TIP][sample.active_frame_range[0]:sample.active_frame_range[1],
              1]) < inter_eye_distance * 2
        cnt = 0
        cnt_max = 0
        for x in temp:
            if cnt:
                cnt += 1
            else:
                cnt_max = cnt
                cnt = 0

        if 'side_by_side' in signwriting_features_list:
            feat.append(np.array(cnt_max > 10).astype(np.float))
        temp = calc_distances(coords['keypoints']['hand_left'][mp_holistic.HandLandmark.INDEX_FINGER_TIP],
                              coords['keypoints']['hand_right'][mp_holistic.HandLandmark.INDEX_FINGER_TIP])
        if 'contacting' in signwriting_features_list:
            feat.append(np.any(temp < inter_eye_distance).astype(np.float))
    return np.array(feat).astype(np.float)

def myplot(coords):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])

    x_coords = []
    y_coords = []
    for x in range(16):
        x_coords.append(coords['face']['face_border_' + format(x)][0, 0])
        y_coords.append(600 - coords['face']['face_border_' + format(x)][0, 1])

    for x in range(51, 68):
        x_coords.append(coords['face']['mouth_' + format(x)][0, 0])
        y_coords.append(600 - coords['face']['mouth_' + format(x)][0, 1])

    for x in range(27, 35):
        x_coords.append(coords['face']['nose_' + format(x)][0, 0])
        y_coords.append(600 - coords['face']['nose_' + format(x)][0, 1])

    ax.plot(x_coords, y_coords, 'r*')
    x_coords = []
    y_coords = []
    x_coords.append(coords['face']['face_border_' + format(0)][0, 0])
    y_coords.append(600 - coords['face']['face_border_' + format(0)][0, 1] )
    x_coords.append(coords['hand_left']['lunate_bone'][0, 0])
    y_coords.append(600 - coords['hand_left']['lunate_bone'][0, 1])
    x_coords.append(coords['hand_left']['index_finger_5'][0, 0])
    y_coords.append(600 - coords['hand_left']['index_finger_5'][0, 1])
    x_coords.append(coords['face']['nose_' + format(33)][0, 0])
    y_coords.append(600 - coords['face']['nose_' + format(33)][0, 1])
    x_coords.append(coords['face']['left_eyebrow_' + format(22)][0, 0])
    y_coords.append(600 - coords['face']['left_eyebrow_' + format(22)][0, 1])
    ax.plot(x_coords, y_coords, 'b.')
    x_coords = []
    y_coords = []
    x_coords.append(coords['hand_left']['index_finger_8'][0, 0])
    y_coords.append(600 - coords['hand_left']['index_finger_8'][0, 1])
    x_coords.append(coords['face']['nose_' + format(27)][0, 0])
    y_coords.append(600 - coords['face']['nose_' + format(27)][0, 1])
    x_coords.append(coords['face']['nose_' + format(27)][0, 0])
    y_coords.append(600 - coords['face']['nose_' + format(27)][0, 1])
    x_coords.append(coords['face']['right_eyebrow_' + format(21)][0, 0])
    y_coords.append(600 - coords['face']['right_eyebrow_' + format(21)][0, 1])
    x_coords.append(((coords['face']['left_eyebrow_22'] + coords['face']['right_eyebrow_21']) / 2)[0, 0])
    y_coords.append(600 - ((coords['face']['left_eyebrow_22'] + coords['face']['right_eyebrow_21']) / 2)[0, 1])
    x_coords.append(coords['face']['mouth_' + format(66)][0, 0])
    y_coords.append(600 - coords['face']['mouth_' + format(66)][0, 1])
    ax.plot(x_coords, y_coords, 'c.')
    # ax.set(xlim=(0, 1200), ylim=(0, 600))

    plt.show()


def calcA(args,preloaded_data):
    d_c = len(args.combined_vocabulary_mapping)
    d_a = len(list(preloaded_data['bsign22k'].values())[0])

    A = np.zeros((d_c+d_a,d_c+d_a), dtype= np.float )
    A_calc_count = np.zeros((d_c+d_a,d_c+d_a), dtype= np.int )
    for ds in args.dataset_names:
        for sm in range(len(args.datasets[ds].samples)):
            sample = args.datasets[ds].samples[sm]
            if sample.description in list(args.combined_vocabulary_class_id.keys()):
                feature = preloaded_data[ds]
                class_id = args.combined_vocabulary_class_id[sample.description]
                class_mapping = args.combined_vocabulary_mapping[class_id]
                A[class_mapping,d_c:] += feature[sample.sign_id]
                A_calc_count[class_mapping,d_c:] +=1
                A[d_c:,class_mapping] += feature[sample.sign_id]
                A_calc_count[d_c:,class_mapping] +=1

    A = A / (A_calc_count + np.finfo(float).eps)
    np.fill_diagonal(A,1)






    return A





def extract_signwriting_features(args, preloaded_data):
    preloaded_data['signwriting_features'] = {}
    args.signwriting_features_list = ['lh_touch_upper_face', 'lh_touch_nose', 'lh_touch_left_ear', 'lh_touch_right_ear',
                                      'lh_touch_left_eye', 'lh_touch_right_eye', 'lh_touch_left_cheek',
                                      'lh_touch_right_cheek','lh_touch_mouth', 'lh_touch_chin', 'lh_touch_neck',
                                      'lh_touch_left_shoulder','lh_touch_right_shoulder', 'lh_touch_chest', 'lh_touch_neck',
                                      'rh_touch_upper_face', 'rh_touch_nose', 'rh_touch_left_ear', 'rh_touch_right_ear',
                                      'rh_touch_left_eye', 'rh_touch_right_eye', 'rh_touch_left_cheek',
                                      'rh_touch_right_cheek', 'rh_touch_mouth', 'rh_touch_chin', 'rh_touch_neck',
                                      'rh_touch_left_shoulder', 'rh_touch_right_shoulder', 'rh_touch_chest',
                                      'rh_touch_neck',
                                      'together','apart','Circular','Repetitive','up_down','side_to_side',
                                      'Two Hand','side_by_side','contacting']
    for ds in args.datasets.keys():
        preloaded_data['signwriting_features'][ds] = {}
        print('Extracting Features for ' + ds)
        for sm in tqdm.tqdm(range(len(args.datasets[ds].samples))):
            sample = args.datasets[ds].samples[sm]
            coords = preloaded_data[args.coordinate_detection_library][ds][sample.sign_id]
            feat_mv = movement_feature(sample, coords,args.signwriting_features_list, args)
            feat_dom = dominant_hand_movement(sample, coords,args.signwriting_features_list, args)
            feat_morph = np.array(sample.gesture_properties['Mono'] if sample.dataset == 'bsign22k' else 0).astype(np.float)
            feat_signer = np.zeros(43+6,np.float)
            feat_signer[int(sample.signer_id) - 2 if sample.dataset == 'bsign22k' else int(sample.signer_id) + 6 ] = 1
            features  = np.concatenate([feat_mv, feat_dom, np.expand_dims(feat_morph,0), feat_signer],axis=0)
            preloaded_data['signwriting_features'][ds][sample.sign_id] = features

    A = calcA(args,preloaded_data['signwriting_features'])
    preloaded_data['A'] = A


    os.makedirs('configuration/preloaded_data',exist_ok=True)
    with open('configuration/preloaded_data/signwriting_features.pickle', 'wb') as handle:
        pickle.dump(preloaded_data['signwriting_features'], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('configuration/preloaded_data/A.pickle', 'wb') as handle:
        pickle.dump(preloaded_data['A'], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return args


def preload_signwriting_coordinates():
    with open('configuration/preloaded_data/signwriting_features.pickle', 'rb') as handle:
        data = pickle.load(handle)
    with open('configuration/preloaded_data/A.pickle', 'rb') as handle2:
        A = pickle.load(handle2)
        return data, A