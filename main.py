import torch
import wandb
from config import config_definitions
from utility.functions import *
from data.preprocessing.dataset_utility import split_datasets, preload_openpose_coordinates, detect_active_frames, preload_mediapipe_coordinates
from data.preprocessing.extract_signwriting_features import preload_signwriting_coordinates
from training.train_GCN import run_training_scripts

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver' if not is_windows() else 'spawn')

    args = config_definitions()
    args = split_datasets(args)
    if args.use_wandb:
        args.wandb = wandb
        config =  { key: args.__dict__[key] for key in args.__dict__.keys() if type(args.__dict__[key]) == str or  type(args.__dict__[key]) == int or type(args.__dict__[key]) == float or type(args.__dict__[key]) == bool or type(args.__dict__[key]) == None or type(args.__dict__[key]) == list}

        args.wandb.init(project="domain_adaptation_sl", entity="pilab", name=args.wandb_name, config=config)
        #args.wandb.config.update(args)


    if args.model_type in ['GCN', 'SSTCN']:
        if args.coordinate_detection_library == 'openpose':
            preloaded_data = {'openpose': preload_openpose_coordinates(args)}
        elif args.coordinate_detection_library == 'mediapipe':
            preloaded_data = {'mediapipe': preload_mediapipe_coordinates(args)}
        else:
            assert False, 'Unknown coordinate_detection_library'
        args = detect_active_frames(args, preloaded_data[args.coordinate_detection_library])
        if args.use_multilabel != 'no':
            preloaded_data['signwriting'],preloaded_data['A'] = preload_signwriting_coordinates()

        run_training_scripts(args, preloaded_data)
    else:
        run_training_scripts(args)
