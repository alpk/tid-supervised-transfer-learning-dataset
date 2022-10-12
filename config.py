import ArgConfigParse
import socket
import os
import glob
import json

from pathlib import Path
from data.preprocessing.dataset_utility import DatasetInformation

_hostname_prefixes = ('login', 'sn01', 'dgx01', 'pilab-vision', 'akya', 'barbun')
_datasets = ['AUTSL', 'bsign22k', 'csl', 'msasl', 'bsl1000']
_dataset_subsets = ['AUTSL_train_shared','AUTSL_train_whole','AUTSL_val_shared','AUTSL_val_whole',
                    'bsign22k_train_shared','bsign22k_train_whole','bsign22k_val_shared','bsign22k_val_whole',
                    'csl_train', 'csl_val'
                    ]

_models = ['GCN', 'dsGCN'  'mc3_18', 'r3d_18', 'r2plus1d_18', 'SSTCN']


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        assert False, 'Boolean value expected.'


def config_definitions():
    # Set All Local Paths here
    # create the argument parser object
    if any(substring in socket.gethostname() for substring in ['pilab-vision']):
        machine_name = 'pilab-vision'
    elif any(substring in socket.gethostname() for substring in ['login', 'sn01', 'dgx01']):
        machine_name = 'tam-cluster'
    elif any(substring in socket.gethostname() for substring in ['akya', 'barbun', 'levrek', 'palamut', 'hamsi']):
        machine_name = 'truba-cluster'
    elif any(substring in socket.gethostname() for substring in ['gesture']):
        machine_name = 'gesture'
    else:
        assert False, 'add paths in config'

    config_bsign = ArgConfigParse.ConfigFile(['configuration/ds/bsign22k.ini'])
    config_bsign.parse_config()
    config_autsl = ArgConfigParse.ConfigFile(['configuration/ds/autsl.ini'])
    config_autsl.parse_config()
    config_csl = ArgConfigParse.ConfigFile(['configuration/ds/csl.ini'])
    config_csl.parse_config()
    bsign22k = DatasetInformation(name='bsign22k',
                                  videos=config_bsign.config_dict[machine_name]['videos'],
                                  class_files=config_bsign.config_dict[machine_name]['class_files'],
                                  frames_root=config_bsign.config_dict[machine_name]['frames_root'],
                                  kinect_root=config_bsign.config_dict[machine_name]['kinect_root'],
                                  openpose_root=config_bsign.config_dict[machine_name]['openpose_root'],
                                  mediapipe_root=config_bsign.config_dict[machine_name]['mediapipe_root'],
                                  dataset_mean=json.loads(config_bsign.config_dict['Main']['dataset_mean']),
                                  dataset_std=json.loads(config_bsign.config_dict['Main']['dataset_std']),
                                  class_attribute_file= config_bsign.config_dict[machine_name]['class_files'])

    AUTSL = DatasetInformation(name='AUTSL',
                               videos=config_autsl.config_dict[machine_name]['videos'],
                               class_files=config_autsl.config_dict[machine_name]['class_files'].split(','),
                               frames_root=config_autsl.config_dict[machine_name]['frames_root'],
                               kinect_root=config_autsl.config_dict[machine_name]['kinect_root'],
                               openpose_root=config_autsl.config_dict[machine_name]['openpose_root'],
                               mediapipe_root=config_autsl.config_dict[machine_name]['mediapipe_root'],
                               dataset_mean=json.loads(config_autsl.config_dict['Main']['dataset_mean']),
                               dataset_std=json.loads(config_autsl.config_dict['Main']['dataset_std']),
                               class_attribute_file= config_autsl.config_dict[machine_name]['class_attribute_file'])

    CSL = DatasetInformation(name='CSL',
                               videos=config_autsl.config_dict[machine_name]['videos'],
                               class_files=config_autsl.config_dict[machine_name]['class_files'].split(','),
                               frames_root=config_autsl.config_dict[machine_name]['frames_root'],
                               kinect_root=config_autsl.config_dict[machine_name]['kinect_root'],
                               openpose_root=config_autsl.config_dict[machine_name]['openpose_root'],
                               mediapipe_root=config_bsign.config_dict[machine_name]['mediapipe_root'],
                               dataset_mean=json.loads(config_autsl.config_dict['Main']['dataset_mean']),
                               dataset_std=json.loads(config_autsl.config_dict['Main']['dataset_std']),
                               class_attribute_file= config_autsl.config_dict[machine_name]['class_attribute_file'])

    arg_parser = ArgConfigParse.CmdArgs()

    # Program options on what to run
    arg_parser.add_argument('--experiment_notes', type=str, default='autsl finetuing')
    arg_parser.add_argument('--extract_frames', type=bool, default=False)
    arg_parser.add_argument('--run_dataset_parse', type=bool, default=True)
    arg_parser.add_argument('--transfer_train_source', nargs='+', default=['AUTSL_train_shared'])
    arg_parser.add_argument('--transfer_train_target', nargs='*', default=['AUTSL_train_shared'])#default=['bsign22k_train_shared'])
    arg_parser.add_argument('--transfer_validation', nargs='+', default=['AUTSL_val_shared'])#default=['bsign22k_val_shared'])
    arg_parser.add_argument('--transfer_method', nargs='+', default='single_target', choices=['single_target', 'dsGCN', 'combined', 'mcc', 'DANN', 'mdd', 'jan', 'mcc_DANN', 'mcc_jan'])
    arg_parser.add_argument('--use_multilabel', nargs='+', default='graph_multilabel', choices=['no', 'baseline', 'graph_multilabel'])

    arg_parser.add_argument('--phase_list', nargs='+', default=['train', 'val'])

    #WANDB part
    arg_parser.add_argument('--use_wandb', type=bool, default=False)
    arg_parser.add_argument('--wandb_name', type=str, default=None)
    autsl_indexes = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 15, 17, 19, 20, 21, 22, 23, 24, 26, 28, 29, 31, 32, 33, 36, 37,
                     38, 40, 41, 42]
    bsign_indexes = [2, 3, 4, 5, 6, 7]
    arg_parser.add_argument('--bsign_user_index', nargs='+', default=[2])#[2,5,6,7])
    arg_parser.add_argument('--autsl_user_index', nargs='+', default=["all"])
    arg_parser.add_argument('--csl_user_index', nargs='+', default=map(str, list(range(1,45))))

    arg_parser.add_argument('--csl_val_user_index', nargs='+', default=map(str, list(range(45,51))))
    
    arg_parser.add_argument('--bsign_val_user_index', nargs='+', default=[3,4])

    arg_parser.add_argument('--model_type', type=str, default='GCN', choices=_models)
    arg_parser.add_argument('--dataset_names', nargs='*', default=['bsign22k', 'AUTSL'], choices=_datasets)
    arg_parser.parse_args()
    datasets = {}
    if 'bsign22k' in arg_parser.options.dataset_names:
        datasets['bsign22k'] = bsign22k
    if 'AUTSL' in arg_parser.options.dataset_names:
        datasets['AUTSL'] = AUTSL

    arg_parser.add_argument('--datasets', type=dict, default=datasets,
                            choices=_datasets)  # , , 'AUTSL': autsl, 'bsign22k': bsign22k},

    arg_parser.add_argument('--max_num_classes', type=int, default=-1)
    arg_parser.add_argument('--iterations_per_epoch', type=int, default=-1)
    arg_parser.add_argument('--display_batch_progress', action='store_true')
    arg_parser.add_argument('--num_workers', type=int, default=0)
    arg_parser.add_argument('--device_ids', nargs='+', type=int, default=[])

    arg_parser.parse_args()
    if 'SSTCN' in arg_parser.nested_opts_dict['__cmd_line']['model_type']:
        config_mdl = ArgConfigParse.ConfigFile(['configuration/algorithms/sstcn.ini'])
    elif 'GCN' in arg_parser.nested_opts_dict['__cmd_line']['model_type']:
        config_mdl = ArgConfigParse.ConfigFile(['configuration/algorithms/gcn.ini'])
    elif 'dsGCN' in arg_parser.nested_opts_dict['__cmd_line']['model_type']:
        config_mdl = ArgConfigParse.ConfigFile(['configuration/algorithms/gcn.ini'])
    elif 'HCN' in arg_parser.nested_opts_dict['__cmd_line']['model_type']:
        config_mdl = ArgConfigParse.ConfigFile(['configuration/algorithms/hcn.ini'])
    elif 'ataf' in arg_parser.nested_opts_dict['__cmd_line']['model_type']:
        config_mdl = ArgConfigParse.ConfigFile(['configuration/algorithms/taf.ini'])
    elif 'r2plus1d_18' in arg_parser.nested_opts_dict['__cmd_line']['model_type']:
        config_mdl = ArgConfigParse.ConfigFile(['configuration/algorithms/r2plus1d_18.ini'])
    elif 'mc3_18' in arg_parser.nested_opts_dict['__cmd_line']['model_type']:
        config_mdl = ArgConfigParse.ConfigFile(['configuration/algorithms/mc3_18.ini'])
    else:
        assert False, 'Model config file missing'

    config_model = config_mdl.parse_config()

    arg_parser.add_argument('--pretrained_model', type=str, default=config_model['Main']['pretrained_model'])
    arg_parser.add_argument('--num_epochs', type=int, default=config_model['Main']['num_epochs'])

    # Training Parameters
    arg_parser.add_argument('--batch_size', type=int, default=config_model['Main']['batch_size'])
    arg_parser.add_argument('--learning_rate', type=float, default=config_model['Main']['learning_rate'])
    arg_parser.add_argument('--weight_decay', type=float, default=config_model['Main']['weight_decay'])
    arg_parser.add_argument('--scheduler', type=str, default=config_model['Main']['scheduler'])
    arg_parser.add_argument('--optimizer', type=str, default=config_model['Main']['optimizer'])
    arg_parser.add_argument('--warmup', type=bool, default=str2bool(config_model['Main']['warmup']))

    arg_parser.add_argument('--metrics', nargs='*', default=['acc1', 'acc5'])
    arg_parser.add_argument('--freeze_layer_names', nargs='+', default=[config_model['Main']['freeze_layer_names']])
    arg_parser.add_argument('--freeze_layer_epochs', nargs='+', type=int, default=[0])
    arg_parser.add_argument('--input_image_crop_mode', type=str, default='none',
                            choices=['none', 'hands', 'hands_body', 'hands_body_face'])
    arg_parser.add_argument('--accumulate_loss_over_batches', type=int, default=1)
    arg_parser.add_argument('--coordinate_detection_library', type=str, default='openpose',choices=['openpose','mediapipe'])

    arg_parser.add_argument('--keyframe_methods', type=str, default=config_model['Main']['keyframe_methods'],
                            choices=['none', 'hs_density_cluster', 'finger_density_cluster',
                                     'hs_varlen_density_cluster',
                                     'ent_density_cluster', 'deep_CTW', 'Task_Cluster_CTW'])
    arg_parser.add_argument('--no_of_static_keyframes', type=int,
                            default=config_model['Main']['no_of_static_keyframes'])  # -1 for dynamic
    arg_parser.add_argument('--alignment_method', type=str, default=config_model['Main']['alignment_method'],
                            choices=['none', 'dtw_kf', 'dctw_kf', 'ttn_kf', 'ttn_sequence'])

    # coordinate augmentation
    arg_parser.add_argument('--temporal_sampling', type=str, default=config_model['Main']['temporal_sampling'],
                            choices=['linear_sampling', 'random_linear_sampling', 'pad_zeros'])
    arg_parser.add_argument('--random_choose', type=bool,
                            default=str2bool(config_model['Main']['random_choose']))  # Not implemented
    arg_parser.add_argument('--random_shift', type=bool, default=str2bool(config_model['Main']['random_shift']))
    arg_parser.add_argument('--normalization', type=bool, default=str2bool(config_model['Main']['normalization']))
    arg_parser.add_argument('--random_mirror', type=bool, default=str2bool(config_model['Main']['random_mirror']))

    # model params
    arg_parser.add_argument('--input_size', type=int,
                            default=config_model['Main']['input_size'])  # {'S':160, 'M':224, 'XL':312}
    arg_parser.add_argument('--downsampling_factor', type=int, default=config_model['Main']['downsampling_factor'])
    arg_parser.add_argument('--downsampling_factor_taf_calculate', type=int,
                            default=config_model['Main']['downsampling_factor_taf_calculate'])
    arg_parser.add_argument('--downsampling_factor_heatmap', type=int,
                            default=config_model['Main']['downsampling_factor_heatmap'])
    arg_parser.add_argument('--clip_length', type=int, default=config_model['Main']['clip_length'])  # 20 S 40 M 80 XL
    arg_parser.add_argument('--output_type', type=str, default=config_model['Main']['output_type'],
                            choices=['cls', 'coord'])  # coord
    arg_parser.add_argument('--input_type', type=str, default=config_model['Main']['input_type'],
                            choices=['frame', 'ataf'])  # coord
    arg_parser.add_argument('--dropout', type=float, default=config_model['Main']['dropout'])
    arg_parser.add_argument('--block_size', type=int, default=config_model['Main']['block_size'])
    arg_parser.add_argument('--filter_size', type=int, default=config_model['Main']['filter_size'])
    arg_parser.add_argument('--convolution_size', type=int, default=config_model['Main']['convolution_size'])
    arg_parser.add_argument('--joint_groups', type=int, default=config_model['Main']['joint_groups'])
    arg_parser.add_argument('--find_optimal_lr', type=bool, default=False)

    arg_parser.parse_args()
    args = arg_parser.options


    args.using_nonshared_gestures = any(
        [x for x in args.transfer_train_source + args.transfer_train_target + args.transfer_validation if 'whole' in x])
    args.using_autsl_gestures = any(
        [x for x in args.transfer_train_source + args.transfer_train_target + args.transfer_validation if 'AUTSL' in x])
    return args


if __name__ == '__main__':
    parser = config_definitions()