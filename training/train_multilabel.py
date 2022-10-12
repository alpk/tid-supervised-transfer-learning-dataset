import torch.utils.data
import torch.nn.functional as Fn

from models.video_resnet import r2plus1d_18, mc3_18, r3d_18
from models.dsGCN import Model as dsGCN
from models.GCN import Model
from models.DANN import DANN
from models.jan import JAN, JointMultipleKernelMaximumMeanDiscrepancy, Theta, GaussianKernel
from models.mdd import MDDLoss, MDD
from utility.functions import *
from utility.sequence_randaugment import RandAugment
from data.preprocessing.transforms import *
from data.sign_language_dataset import SignLanguageDataset
from data.preprocessing.transforms_coordinates import Normalization, RandomMirror, RandomShift, RandomChoose
from torch.utils.data import DataLoader
from models.mcc import MinimumClassConfusionLoss
from torch.cuda.amp import GradScaler, autocast
from utility.meter import ProgressMeter
from itertools import cycle

import time
import gc
import pytorch_warmup as warmup

def train_multilabel(args, preloaded_data= None):
    # Initialize experiment folder
    experiment_folder = initialize_experiment_folder(args)
    # Write experiment Config
    write_experiment_config_file(args, experiment_folder)

    # Initialize loggers
    sw, log = initialize_loggers(experiment_path=experiment_folder, splits=args.phase_list, metrics=args.metrics)

    # Initialize loggers
    fig = initialize_figure_drawers(experiment_path=experiment_folder, splits=args.phase_list, metrics=args.metrics)

    # Initialize device
    device = get_device()  #'cpu'#