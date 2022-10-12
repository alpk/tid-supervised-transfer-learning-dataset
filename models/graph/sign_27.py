import sys

sys.path.extend(['../'])
from models.graph import tools


class Graph:
    def __init__(self, labeling_mode='spatial', joint_groups=[], num_point=25, coordinate_type = 'openpose'):
        self_link = [(i, i) for i in range(num_point)]
        joint_names = [x[0]+'_'+ str(x[1]) for x in joint_groups]
        if coordinate_type == 'openpose':
            connections = [('pose_nose', 'pose_neck'),
                           ('pose_nose', 'pose_left_eye'),
                           ('pose_nose', 'pose_right_eye'),
                            ('pose_neck', 'pose_left_shoulder'),
                            ('pose_left_shoulder', 'pose_left_elbow'),
                            ('pose_left_elbow', 'pose_left_wrist'),
                            ('pose_neck', 'pose_right_shoulder'),
                            ('pose_right_shoulder', 'pose_right_elbow'),
                            ('pose_left_elbow', 'hand_left_lunate_bone'),
                            ('pose_left_wrist', 'hand_left_lunate_bone'),
                            ('hand_left_lunate_bone', 'hand_left_thumb_1'),
                           ('hand_left_thumb_1', 'hand_left_thumb_4'),
                           ('hand_left_lunate_bone', 'hand_left_index_finger_5'),
                           ('hand_left_index_finger_5', 'hand_left_index_finger_8'),
                           ('hand_left_lunate_bone', 'hand_left_middle_finger_9'),
                           ('hand_left_middle_finger_9', 'hand_left_middle_finger_12'),
                           ('hand_left_lunate_bone', 'hand_left_ring_finger_13'),
                           ('hand_left_ring_finger_13', 'hand_left_ring_finger_16'),
                           ('hand_left_lunate_bone', 'hand_left_little_finger_17'),
                           ('hand_left_little_finger_17', 'hand_left_little_finger_20'),
                           ('pose_right_elbow', 'hand_right_lunate_bone'),
                           ('pose_right_wrist', 'hand_right_lunate_bone'),
                            ('hand_right_lunate_bone', 'hand_right_thumb_1'),
                           ('hand_right_thumb_1', 'hand_right_thumb_4'),
                           ('hand_right_lunate_bone', 'hand_right_index_finger_5'),
                           ('hand_right_index_finger_5', 'hand_right_index_finger_8'),
                           ('hand_right_lunate_bone', 'hand_right_middle_finger_9'),
                           ('hand_right_middle_finger_9', 'hand_right_middle_finger_12'),
                           ('hand_right_lunate_bone', 'hand_right_ring_finger_13'),
                           ('hand_right_ring_finger_13', 'hand_right_ring_finger_16'),
                           ('hand_right_lunate_bone', 'hand_right_little_finger_17'),
                           ('hand_right_little_finger_17', 'hand_right_little_finger_20'),
                           ]
        elif coordinate_type == 'mediapipe':
            connections = [('pose_PoseLandmark.NOSE', 'pose_PoseLandmark.MOUTH_LEFT'),
                           ('pose_PoseLandmark.NOSE', 'pose_PoseLandmark.LEFT_EYE'),
                           ('pose_PoseLandmark.NOSE', 'pose_PoseLandmark.RIGHT_EYE'),

                           ('pose_PoseLandmark.NOSE', 'pose_PoseLandmark.LEFT_SHOULDER'),
                           ('pose_PoseLandmark.LEFT_SHOULDER', 'pose_PoseLandmark.LEFT_ELBOW'),
                           ('pose_PoseLandmark.LEFT_ELBOW', 'pose_PoseLandmark.LEFT_WRIST'),
                           ('pose_PoseLandmark.LEFT_ELBOW', 'hand_left_HandLandmark.WRIST'),
                           ('pose_PoseLandmark.LEFT_WRIST', 'hand_left_HandLandmark.WRIST'),
                           ('hand_left_HandLandmark.WRIST', 'hand_left_HandLandmark.THUMB_CMC'),

                           ('hand_left_HandLandmark.THUMB_CMC', 'hand_left_HandLandmark.THUMB_TIP'),
                           ('hand_left_HandLandmark.WRIST', 'hand_left_HandLandmark.INDEX_FINGER_MCP'),
                           ('hand_left_HandLandmark.INDEX_FINGER_MCP', 'hand_left_HandLandmark.INDEX_FINGER_TIP'),
                           ('hand_left_HandLandmark.WRIST', 'hand_left_HandLandmark.MIDDLE_FINGER_MCP'),
                           ('hand_left_HandLandmark.MIDDLE_FINGER_MCP', 'hand_left_HandLandmark.MIDDLE_FINGER_TIP'),
                           ('hand_left_HandLandmark.WRIST', 'hand_left_HandLandmark.RING_FINGER_MCP'),
                           ('hand_left_HandLandmark.RING_FINGER_MCP', 'hand_left_HandLandmark.RING_FINGER_TIP'),
                           ('hand_left_HandLandmark.WRIST', 'hand_left_HandLandmark.PINKY_MCP'),
                           ('hand_left_HandLandmark.PINKY_MCP', 'hand_left_HandLandmark.PINKY_TIP'),

                           ('pose_PoseLandmark.NOSE', 'pose_PoseLandmark.RIGHT_SHOULDER'),
                           ('pose_PoseLandmark.RIGHT_SHOULDER', 'pose_PoseLandmark.RIGHT_ELBOW'),
                           ('pose_PoseLandmark.RIGHT_ELBOW', 'pose_PoseLandmark.RIGHT_WRIST'),
                           ('pose_PoseLandmark.RIGHT_ELBOW', 'hand_right_HandLandmark.WRIST'),
                           ('pose_PoseLandmark.RIGHT_WRIST', 'hand_right_HandLandmark.WRIST'),

                           ('hand_right_HandLandmark.WRIST', 'hand_right_HandLandmark.THUMB_CMC'),
                           ('hand_right_HandLandmark.THUMB_CMC', 'hand_right_HandLandmark.THUMB_TIP'),
                           ('hand_right_HandLandmark.WRIST', 'hand_right_HandLandmark.INDEX_FINGER_MCP'),
                           ('hand_right_HandLandmark.INDEX_FINGER_MCP', 'hand_right_HandLandmark.INDEX_FINGER_TIP'),
                           ('hand_right_HandLandmark.WRIST', 'hand_right_HandLandmark.MIDDLE_FINGER_MCP'),
                           ('hand_right_HandLandmark.MIDDLE_FINGER_MCP', 'hand_right_HandLandmark.MIDDLE_FINGER_TIP'),
                           ('hand_right_HandLandmark.WRIST', 'hand_right_HandLandmark.RING_FINGER_MCP'),
                           ('hand_right_HandLandmark.RING_FINGER_MCP', 'hand_right_HandLandmark.RING_FINGER_TIP'),
                           ('hand_right_HandLandmark.WRIST', 'hand_right_HandLandmark.PINKY_MCP'),
                           ('hand_right_HandLandmark.PINKY_MCP', 'hand_right_HandLandmark.PINKY_TIP'),
                           ]
        inward_ori_index = []
        for cn in range(len(connections)):
            if connections[cn][0] in joint_names and connections[cn][1] in joint_names:
                inward_ori_index.append((joint_names.index( connections[cn][0]),joint_names.index( connections[cn][1])))
            else:
                print(connections[cn][0], connections[cn][1])



        inward = [(i, j) for (i, j) in inward_ori_index]
        outward = [(j, i) for (i, j) in inward]
        neighbor = inward + outward

        self.num_node = num_point
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)