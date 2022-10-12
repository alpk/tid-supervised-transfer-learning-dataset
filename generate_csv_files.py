from config import config_definitions
from utility.functions import *
from data.preprocessing.dataset_utility import split_datasets, preload_openpose_coordinates, detect_active_frames, preload_mediapipe_coordinates
from data.preprocessing.extract_signwriting_features import extract_signwriting_features, preload_signwriting_coordinates

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver' if not is_windows() else 'spawn')

    args = config_definitions()
    args = split_datasets(args)
    if args.coordinate_detection_library == 'openpose':
        preloaded_data = {'openpose': preload_openpose_coordinates(args)}
    elif args.coordinate_detection_library == 'mediapipe':
        preloaded_data = {'mediapipe': preload_mediapipe_coordinates(args)}
    else:
        assert False, 'Unknown coordinate_detection_library'
    args = detect_active_frames(args, preloaded_data[args.coordinate_detection_library])

    args = extract_signwriting_features(args, preloaded_data)
    preloaded_data['signwriting'], preloaded_data['A']  = preload_signwriting_coordinates()

    # fix class ids
    AUTSL_mapping_easy = {1: 4, 8: 12, 9: 14, 14: 30, 15: 35, 20: 49, 28: 70, 29: 72, 35: 108, 64: 188, 154: 197, 76: 213,
                     77: 214, 83: 233, 94: 245, 100: 272, 101: 278, 116: 279, 103: 284, 117: 323, 122: 329, 128: 339,
                     125: 346, 151: 401, 163: 410, 165: 422, 172: 434, 176: 436, 193: 490, 204: 504, 205: 518, 206: 527,
                     211: 533, 215: 540, 217: 545, 221: 552, 222: 553, 224: 557, 22: 561, 209: 586}


    AUTSL_mapping = {1: 4, 8: 12, 9: 14, 11: 15, 14: 30, 15: 35, 20: 49, 22: 561, 28: 70, 29: 72, 35: 108, 42: 122,
                         52: 150, 58: 656, 61: 175, 62: 375, 64: 188, 66: 190, 73: 204, 76: 213, 77: 214, 82: 230,
                         83: 233, 94: 245, 96: 228, 100: 272, 101: 278, 103: 284, 112: 369, 116: 279, 117: 323,
                         122: 329, 125: 346, 128: 339, 141: 373, 144: 382, 151: 401, 154: 197, 163: 410, 165: 422,
                         172: 434, 176: 436, 177: 1, 183: 464, 188: 484, 193: 490, 200: 498, 204: 504, 205: 518,
                         206: 527, 209: 586, 211: 534, 215: 540, 217: 545, 221: 552, 222: 553, 224: 557}

    AUTSL_mapping = {int(v): k for k, v in AUTSL_mapping.items()}
    bsign_mapping = {x: x + len(args.datasets['AUTSL'].vocabulary) for x in
                     range(len(args.datasets['bsign22k'].vocabulary))}
    for x in AUTSL_mapping.keys():
        bsign_mapping[x] = AUTSL_mapping[x]

    os.makedirs('./configuration/json/', exist_ok=True)
    for ds in ['bsign22k', 'AUTSL']:
        for ph in ['train', 'val']:
            for subset in ['whole', 'shared']:
                selected_samples = [x for x in args.datasets[ds].samples if x.dataset == ds]
                selected_samples = [x for x in selected_samples if x.phase == ph]
                if subset == 'shared':
                    buf = []
                    for x in selected_samples:
                        if x.dataset == 'AUTSL':
                            if x.class_id in list(AUTSL_mapping.values()):
                                buf.append(x)
                        elif x.gesture_properties['Chalearn Corresponding'] != '':
                            buf.append(x)
                    selected_samples = buf
                with open('./configuration/json/' + ds + '_' + ph + '_' + subset + '.csv', 'w',
                          newline='') as output_file:
                    dict_writer = csv.DictWriter(output_file, ['frames_path', 'openpose_path', 'class_id', 'dataset',
                                                               'image_dims', 'active_frame_range_start',
                                                               'active_frame_range_end', 'description', 'phase',
                                                               'sign_id', 'signer_id', 'ClassNameTr', 'One Hand',
                                                               'Two Hand', 'Circular', 'Repetitive', 'Mono', 'Compound',
                                                               'Chalearn Corresponding'])
                    dict_writer.writeheader()
                    if subset == 'whole' and ds == 'AUTSL':
                        for s in selected_samples:
                            dict_writer.writerow({'frames_path': s.frames_path,
                                                  'openpose_path': s.openpose_path,
                                                  'class_id': s.class_id if s.dataset == 'AUTSL' else bsign_mapping[
                                                      s.class_id],
                                                  'dataset': ds,
                                                  'image_dims': s.image_dims,
                                                  'active_frame_range_start': s.active_frame_range[0],
                                                  'active_frame_range_end': s.active_frame_range[1],
                                                  'description': s.description,
                                                  'sign_id': s.sign_id,
                                                  'signer_id': s.signer_id,
                                                  'ClassNameTr': '', 'One Hand': '', 'Two Hand': '',
                                                  'Circular': '', 'Repetitive': '', 'Mono': '',
                                                  'Compound': '', 'Chalearn Corresponding': s.class_id})
                    else:
                        for s in selected_samples:
                            dict_writer.writerow({'frames_path': s.frames_path,
                                                  'openpose_path': s.openpose_path,
                                                  'class_id': s.class_id if s.dataset == 'AUTSL' else bsign_mapping[
                                                      s.class_id],
                                                  'dataset': ds,
                                                  'image_dims': s.image_dims,
                                                  'active_frame_range_start': s.active_frame_range[0],
                                                  'active_frame_range_end': s.active_frame_range[1],
                                                  'description': s.description,
                                                  'sign_id': s.sign_id,
                                                  'signer_id': s.signer_id,
                                                  'ClassNameTr': s.gesture_properties['ClassNameTr'],
                                                  'One Hand': s.gesture_properties['One Hand'],
                                                  'Two Hand': s.gesture_properties['Two Hand'],
                                                  'Circular': s.gesture_properties['Circular'],
                                                  'Repetitive': s.gesture_properties['Repetitive'],
                                                  'Mono': s.gesture_properties['Mono'],
                                                  'Compound': s.gesture_properties['Compound'],
                                                  'Chalearn Corresponding': s.gesture_properties[
                                                      'Chalearn Corresponding'] if s.dataset == 'bsign22k' else str(s.class_id) })

    print('done')