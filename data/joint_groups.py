import mediapipe as mp

au = {'Head': ['Head'],
      'K_Hands': ['HandLeft', 'HandRight', 'HandTipLeft', 'ThumbLeft', 'HandTipRight', 'ThumbRight'],
      'K_Wrists': ['WristLeft', 'WristRight'],
      'K_Arms': ['ElbowLeft', 'ElbowRight', 'ShoulderLeft', 'ShoulderRight'],
      'O_Head': [['pose', 'nose'], ['pose', 'neck']],
      'O_Eyes': [['pose', 'left_eye'], ['pose', 'right_eye']],
      'O_Wrists': [['pose', 'left_wrist'], ['pose', 'right_wrist']],
      'O_Arms': [['pose', 'left_shoulder'], ['pose', 'left_elbow'], ['pose', 'right_shoulder'],
                 ['pose', 'right_elbow']],
      'O_Wrist': [['hand_left', 'lunate_bone'], ['hand_right', 'lunate_bone']],
      'O_Tips': [['hand_left', 'thumb_4'],
                 ['hand_left', 'index_finger_8'],
                 ['hand_left', 'middle_finger_12'],
                 ['hand_left', 'ring_finger_16'],
                 ['hand_left', 'little_finger_20'],
                 ['hand_right', 'thumb_4'],
                 ['hand_right', 'index_finger_8'],
                 ['hand_right', 'middle_finger_12'],
                 ['hand_right', 'ring_finger_16'],
                 ['hand_right', 'little_finger_20']
                 ],
      'O_Base': [['hand_left', 'thumb_1'],
                 ['hand_left', 'index_finger_5'],
                 ['hand_left', 'middle_finger_9'],
                 ['hand_left', 'ring_finger_13'],
                 ['hand_left', 'little_finger_17'],
                 ['hand_right', 'thumb_1'],
                 ['hand_right', 'index_finger_5'],
                 ['hand_right', 'middle_finger_9'],
                 ['hand_right', 'ring_finger_13'],
                 ['hand_right', 'little_finger_17']],
      'O_FingerJoints': [['hand_left', 'thumb_2'],
                         ['hand_left', 'thumb_3'],
                         ['hand_left', 'index_finger_6'],
                         ['hand_left', 'index_finger_7'],
                         ['hand_left', 'middle_finger_10'],
                         ['hand_left', 'middle_finger_11'],
                         ['hand_left', 'ring_finger_14'],
                         ['hand_left', 'ring_finger_15'],
                         ['hand_left', 'little_finger_18'],
                         ['hand_left', 'little_finger_19'],
                         ['hand_right', 'thumb_2'],
                         ['hand_right', 'thumb_3'],
                         ['hand_right', 'index_finger_6'],
                         ['hand_right', 'index_finger_7'],
                         ['hand_right', 'middle_finger_10'],
                         ['hand_right', 'middle_finger_11'],
                         ['hand_right', 'ring_finger_14'],
                         ['hand_right', 'ring_finger_15'],
                         ['hand_right', 'little_finger_18'],
                         ['hand_right', 'little_finger_19']
                         ]}
mu = {'Head': ['Head'],
      'K_Hands': ['HandLeft', 'HandRight', 'HandTipLeft', 'ThumbLeft', 'HandTipRight', 'ThumbRight'],
      'K_Wrists': ['WristLeft', 'WristRight'],
      'K_Arms': ['ElbowLeft', 'ElbowRight', 'ShoulderLeft', 'ShoulderRight'],
      'O_Head': [['pose', mp.solutions.holistic.PoseLandmark.NOSE], ['pose', mp.solutions.holistic.PoseLandmark.MOUTH_LEFT]],
      'O_Eyes': [['pose', mp.solutions.holistic.PoseLandmark.LEFT_EYE], ['pose', mp.solutions.holistic.PoseLandmark.RIGHT_EYE]],
      'O_Wrists': [['pose',  mp.solutions.holistic.PoseLandmark.LEFT_WRIST], ['pose',  mp.solutions.holistic.PoseLandmark.RIGHT_WRIST]],
      'O_Arms': [['pose', mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER], ['pose', mp.solutions.holistic.PoseLandmark.LEFT_ELBOW], ['pose', mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER],
                 ['pose', mp.solutions.holistic.PoseLandmark.RIGHT_ELBOW]],
      'O_Wrist': [['hand_left', mp.solutions.holistic.HandLandmark.WRIST], ['hand_right', mp.solutions.holistic.HandLandmark.WRIST]],
      'O_Tips': [['hand_left', mp.solutions.holistic.HandLandmark.THUMB_TIP],
                 ['hand_left', mp.solutions.holistic.HandLandmark.INDEX_FINGER_TIP],
                 ['hand_left', mp.solutions.holistic.HandLandmark.MIDDLE_FINGER_TIP],
                 ['hand_left', mp.solutions.holistic.HandLandmark.RING_FINGER_TIP],
                 ['hand_left', mp.solutions.holistic.HandLandmark.PINKY_TIP],
                 ['hand_right', mp.solutions.holistic.HandLandmark.THUMB_TIP],
                 ['hand_right', mp.solutions.holistic.HandLandmark.INDEX_FINGER_TIP],
                 ['hand_right', mp.solutions.holistic.HandLandmark.MIDDLE_FINGER_TIP],
                 ['hand_right', mp.solutions.holistic.HandLandmark.RING_FINGER_TIP],
                 ['hand_right', mp.solutions.holistic.HandLandmark.PINKY_TIP]
                 ],
      'O_Base': [['hand_left', mp.solutions.holistic.HandLandmark.THUMB_CMC],
                 ['hand_left', mp.solutions.holistic.HandLandmark.INDEX_FINGER_MCP],
                 ['hand_left', mp.solutions.holistic.HandLandmark.MIDDLE_FINGER_MCP],
                 ['hand_left', mp.solutions.holistic.HandLandmark.RING_FINGER_MCP],
                 ['hand_left', mp.solutions.holistic.HandLandmark.PINKY_MCP],
                 ['hand_right', mp.solutions.holistic.HandLandmark.THUMB_CMC],
                 ['hand_right', mp.solutions.holistic.HandLandmark.INDEX_FINGER_MCP],
                 ['hand_right', mp.solutions.holistic.HandLandmark.MIDDLE_FINGER_MCP],
                 ['hand_right', mp.solutions.holistic.HandLandmark.RING_FINGER_MCP],
                 ['hand_right', mp.solutions.holistic.HandLandmark.PINKY_MCP]],
      'O_FingerJoints': [['hand_left',  mp.solutions.holistic.HandLandmark.THUMB_MCP],
                         ['hand_left',  mp.solutions.holistic.HandLandmark.THUMB_IP],
                         ['hand_left',  mp.solutions.holistic.HandLandmark.INDEX_FINGER_PIP],
                         ['hand_left',  mp.solutions.holistic.HandLandmark.INDEX_FINGER_DIP],
                         ['hand_left',  mp.solutions.holistic.HandLandmark.MIDDLE_FINGER_PIP],
                         ['hand_left',  mp.solutions.holistic.HandLandmark.MIDDLE_FINGER_DIP],
                         ['hand_left',  mp.solutions.holistic.HandLandmark.RING_FINGER_PIP],
                         ['hand_left',  mp.solutions.holistic.HandLandmark.RING_FINGER_DIP],
                         ['hand_left',  mp.solutions.holistic.HandLandmark.PINKY_PIP],
                         ['hand_left',  mp.solutions.holistic.HandLandmark.PINKY_DIP],
                         ['hand_right', mp.solutions.holistic.HandLandmark.THUMB_MCP],
                         ['hand_right', mp.solutions.holistic.HandLandmark.THUMB_IP],
                         ['hand_right', mp.solutions.holistic.HandLandmark.INDEX_FINGER_PIP],
                         ['hand_right', mp.solutions.holistic.HandLandmark.INDEX_FINGER_DIP],
                         ['hand_right', mp.solutions.holistic.HandLandmark.MIDDLE_FINGER_PIP],
                         ['hand_right', mp.solutions.holistic.HandLandmark.MIDDLE_FINGER_DIP],
                         ['hand_right', mp.solutions.holistic.HandLandmark.RING_FINGER_PIP],
                         ['hand_right', mp.solutions.holistic.HandLandmark.RING_FINGER_DIP],
                         ['hand_right', mp.solutions.holistic.HandLandmark.PINKY_PIP],
                         ['hand_right', mp.solutions.holistic.HandLandmark.PINKY_DIP]]}

def choose_joint_groups(args):
    kinect_joints = []
    openpose_joints = []
    group_joints = []
    if args.coordinate_detection_library == 'openpose':
        if args.joint_groups == 0:
            kinect_joints = au['Head'] + au['K_Arms'] + au['K_Wrists'] + au['K_Hands']
            openpose_joints = au['O_Wrist'] + au['O_Tips']
        elif args.joint_groups == 1:
            kinect_joints = au['K_Wrists']
        elif args.joint_groups == 2:
            kinect_joints = au['Head'] + au['K_Wrists']
        elif args.joint_groups == 3:
            kinect_joints = au['Head'] + au['K_Arms'] + au['K_Wrists']
        elif args.joint_groups == 12:
            kinect_joints = au['Head'] + au['K_Hands'] + au['K_Wrists']
        elif args.joint_groups == 4:
            kinect_joints = au['Head'] + au['K_Hands'] + au['K_Wrists'] + au['K_Arms']
        elif args.joint_groups == 5:
            kinect_joints = au['Head']
            openpose_joints = au['O_Wrist']
        elif args.joint_groups == 6:
            kinect_joints = au['Head']
            openpose_joints = au['O_Wrist'] + au['O_Tips']
        elif args.joint_groups == 7:
            kinect_joints = au['Head']
            openpose_joints = au['O_Wrist'] + au['O_Tips'] + au['O_Base']
        elif args.joint_groups == 8:
            group_joints = [au['Head'] + au['K_Arms'] + au['K_Wrists']]
        elif args.joint_groups == 9:
            group_joints = [au['Head'], au['O_Wrist'] + au['O_Tips']]
        elif args.joint_groups == 10:
            kinect_joints = au['K_Wrists']
            group_joints = [au['Head'] + au['K_Arms'] + au['K_Wrists'], au['O_Wrist'] + au['O_Tips']]
        elif args.joint_groups == 11:
            kinect_joints = au['K_Arms']
            openpose_joints = au['O_Wrist'] + au['O_Tips']
            group_joints = [au['Head'] + au['K_Arms'] + au['K_Wrists'], au['O_Wrist'] + au['O_Tips']]
        elif args.joint_groups == 13:
            kinect_joints = au['Head']
            openpose_joints = au['O_Wrist'] + au['O_Tips']
            group_joints = [au['Head'] + au['K_Arms'], au['O_Wrist'] + au['O_Tips']]
        elif args.joint_groups == 14:
            kinect_joints = au['Head']
            openpose_joints = au['O_Wrist'] + au['O_Tips'] + au['O_Base']
            group_joints = [au['Head'] + au['K_Arms'], au['O_Wrist'] + au['O_Tips']]
        elif args.joint_groups == 15:
            kinect_joints = au['K_Hands']
        elif args.joint_groups == 16:
            kinect_joints = au['Head'] + au['K_Hands']
        elif args.joint_groups == 17:
            kinect_joints = au['Head'] + au['K_Hands'] + au['K_Arms']
        elif args.joint_groups == 18:
            kinect_joints = au['Head'] + au['K_Hands']
            openpose_joints = au['O_Tips']
        elif args.joint_groups == 19:
            kinect_joints = au['Head'] + au['K_Hands']
            openpose_joints = au['O_Tips'] + au['O_Base']
        elif args.joint_groups == 20:
            kinect_joints = au['Head'] + au['K_Hands'] + au['K_Arms']
            openpose_joints = au['O_Tips'] + au['O_Base']
        elif args.joint_groups == 21:
            kinect_joints = au['Head'] + au['K_Hands'] + au['K_Arms']
            openpose_joints = au['O_Tips']
        elif args.joint_groups == 22:
            kinect_joints = au['Head'] + au['K_Hands'] + au['K_Arms']
            openpose_joints = au['O_Base']
        elif args.joint_groups == 23:
            kinect_joints = au['Head'] + au['K_Hands'] + au['K_Arms']
            openpose_joints = au['O_Base'] + au['O_Wrist'] + au['O_Tips']
        elif args.joint_groups == 24:
            kinect_joints = au['Head'] + au['K_Hands'] + au['K_Arms']
            openpose_joints = au['O_Wrist'] + au['O_Tips']
        elif args.joint_groups == 25:
            kinect_joints = au['Head'] + au['K_Arms']
            openpose_joints = au['O_Wrist'] + au['O_Tips']
        elif args.joint_groups == 26:
            openpose_joints = au['O_Head'] + au['O_Arms'] +au['O_Wrist'] + au['O_Tips']
        elif args.joint_groups == 27:
            openpose_joints = au['O_Head'] + au['O_Eyes'] + au['O_Arms'] +au['O_Wrist'] + au['O_Tips']+ au['O_Base']
        elif args.joint_groups == 28:
            openpose_joints = au['O_Head'] + au['O_Arms'] +au['O_Wrist'] + au['O_Base'] + au['O_Tips']
    elif args.coordinate_detection_library == 'mediapipe':

        if args.joint_groups == 27:
            openpose_joints = mu['O_Head'] + mu['O_Arms'] + mu['O_Wrist'] + mu['O_Base'] + mu['O_Tips']
        elif args.joint_groups == 28:
            openpose_joints = mu['O_Head'] + mu['O_Arms'] +mu['O_Wrist'] + mu['O_Base'] + mu['O_Tips']

    return kinect_joints, openpose_joints, group_joints
