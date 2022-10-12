import torch.utils.data

from models.video_resnet import r2plus1d_18, mc3_18, r3d_18
from models.GCN import Model
from models.DANN import DANN
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


def train_C3D(args, preloaded_data= None):
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

    # Initialize dataset and dataloader
    if args.model_type not in ['GCN']:
        num_classes, joint_groups, num_channels, dataloaders = initialize_video_dataloaders(args=args)
    else:
        num_classes, joint_groups, num_channels, dataloaders = initialize_coordinate_dataloaders(args=args, preloaded_data=preloaded_data)

    # Initialize model
    feature_size = 512
    if args.model_type == 'r2plus1d_18':
        model = r2plus1d_18(pretrained=True, dropout_p=args.dropout)
        model.fc = torch.nn.Linear(feature_size, num_classes)
    elif args.model_type == 'mc3_18':
        model = mc3_18(pretrained=True, dropout_p=args.dropout)
        model.fc = torch.nn.Linear(feature_size, num_classes)
    elif args.model_type == 'GCN':
        feature_size = 256
        model = Model(num_class=num_classes, num_point=num_channels,
                      num_person=1, groups=16, in_channels=3, graph='graph.sign_27.Graph',
                      graph_args={'labeling_mode': 'spatial',
                                  'joint_groups': joint_groups,
                                  'num_point': num_channels,
                                  'coordinate_type':args.coordinate_detection_library})

    if args.transfer_method == "DANN":                                
        model = DANN(model, num_classes, feature_size).to(device)    
    elif args.transfer_method == "mdd":
        model = MDD(model, num_classes, feature_size).to(device)
        margin = 4
        mdd_loss_fn = MDDLoss(margin).to(device)
    else:
        pass

    if args.pretrained_model != '':
        model_dict = model.state_dict()
        pretrained_model = torch.load(args.pretrained_model, map_location='cpu')
        pretrained_dict = pretrained_model['model_state_dict']
        try:
            model.load_state_dict(pretrained_dict, strict=True)
            model.load_state_dict(model_dict)
        except:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(pretrained_dict, strict=False)

    model = model.to(device)
    if len(args.device_ids) > 1:
        model = torch.nn.DataParallel(model)
    # Initialize loss function, optimizer, and learning scheduler
    optimizers = {
        'adamw': torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay),
        'sgd': torch.optim.SGD(params=model.parameters(), momentum=0.9, weight_decay=args.weight_decay,
                               lr=args.learning_rate)}
    optimizer = optimizers[args.optimizer]

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    loss_fn.to(device)
    iter_len = get_training_iteration_count(args, dataloaders)

    if args.scheduler in ['one_cycle'] and 'train' in args.phase_list:

        for g in optimizer.param_groups:
            g['lr'] = args.learning_rate * 100
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate,
                                                        steps_per_epoch=iter_len,
                                                        epochs=args.num_epochs, pct_start=0.2)
    elif args.scheduler == 'multi_step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[150, 200], gamma=0.1)
    elif args.scheduler == 'plateu':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                               min_lr=1e-6)

    if args.warmup:
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=1000)

    if args.find_optimal_lr:
        lrf = LearningRateFinder(model, criterion=loss_fn, optimizer=optimizer)
        lrf.fit(args, device, data_loader=dataloaders['train'], steps=200, min_lr=1e-10)
        lrf.plot()
        return

    # epoch loop
    best_metric_loss = float('inf')
    best_metric_acc = float('-inf')

    mcc_loss_fn = MinimumClassConfusionLoss(2.0)

    scaler = GradScaler()
    for epoch in range(0, args.num_epochs):
        is_best = False
        freeze_layers(epoch, model, args.freeze_layer_names, args.freeze_layer_epochs, freeze_all_bn=False, verbose=True)
        for phase in args.phase_list:
            gc.collect()
            if phase == 'train':
                model.train()
                is_best = False
            elif phase in ['val', 'test']:
                model.eval()
                val_iter = iter(dataloaders[phase])
                logger_predictions = create_epoch_predictions_logger(phase, epoch, experiment_folder)
                logger_challenge_predictions = create_epoch_predictions_logger_challenge(phase, epoch, experiment_folder)
                logger_challenge_logits = create_epoch_logits_logger_challenge(phase, epoch, experiment_folder)
                logger_challenge_features = create_epoch_features_logger_challenge(phase, epoch, experiment_folder)
            metrics = initialize_epoch_metrics(phase, args.metrics)
            end_time = time.time()
            # Epoch size is set to larger source or target
            iter_len = get_training_iteration_count(args, dataloaders)
            iter_len = iter_len if phase == 'train' else len(dataloaders['val'])

            progress = ProgressMeter(
                num_batches=iter_len,
                meters=[m for _, m in metrics.items()],
                prefix='Epoch: [{}] {}'.format(epoch, phase)
            )

            for idx in range(iter_len):
                if phase == 'train': # Get Train Data from source and target domain
                    if args.transfer_method != 'single_target':
                        if iter_len >= len(dataloaders['train']):
                            input_s, label_s, name_s, _, sample_id_s = next(iter(dataloaders['train']))
                        else:
                            input_s, label_s, name_s, _, sample_id_s = next(iter(cycle(dataloaders['train'])))
                        if iter_len >= len(dataloaders['target']):
                            input_t, label_t, name_t, _, sample_id_t = next(iter(dataloaders['target']))
                        else:
                            input_t, label_t, name_t, _, sample_id_t = next(iter(cycle(dataloaders['target'])))
                        input = torch.cat((input_s, input_t), dim=0)
                        label = torch.cat((label_s, label_t), dim=0)
                        name = name_s + name_t
                        sample_id = sample_id_s + sample_id_t
                        domain = [len(name_s),len(name_t)]
                    else: # Get Train Data from only target domain
                        input, label, name, signer, sample_id = next(iter(dataloaders['target']))
                        domain = [0, len(name)]
                else: # Get Validation Data
                    input, label, name, signer, sample_id = next(val_iter)
                    domain = [0, len(name)]

                input = input.to(device)
                label = label.to(device)
                metrics['data_time'].update(time.time() - end_time)
                if args.transfer_method == "DANN":
                    alpha = 3 # alpha value of the gradient reversal
                    if args.model_type == 'GCN':
                        output, domain_output, features = model(input.float() / args.input_size, alpha)
                        output = output.squeeze(dim=-1).squeeze(dim=-1)

                    else:
                        with autocast():
                            output, domain_output, features = model(input, alpha)
                
                elif args.transfer_method == "mdd":
                    alpha = 3 # alpha value of the gradient reversal
                    if args.model_type == 'GCN':
                        trade_off = 1.0
                        output, output_adv, features = model(input.float() / args.input_size)
                        output = output.squeeze(dim=-1).squeeze(dim=-1)
                    else:
                        with autocast():
                            output, output_adv, features = model(input)
                
                else:
                    if args.model_type == 'GCN':
                        keep_prob = 1 - args.dropout if phase == "train" else 1.0
                        output, features = model(input.float() / args.input_size, keep_prob)
                        output = output.squeeze(dim=-1).squeeze(dim=-1)

                    else:
                        with autocast():
                            output, features = model(input)                


                logits = output
                output = torch.softmax(output, dim=-1)
                loss = None
                if phase in ['train', 'val']:
                    if args.transfer_method == 'mcc':
                        ce_loss = loss_fn(logits, label)
                        mcc_loss = mcc_loss_fn(logits[domain[0]:,:],)
                        loss = ce_loss + mcc_loss
                    elif args.transfer_method == "DANN":
                        ce_loss = loss_fn(logits, label)
                        domain_loss_s = loss_fn(domain_output[:domain[0]], torch.zeros(domain[0]).long().to(device))
                        domain_loss_t = loss_fn(domain_output[domain[0]:], torch.ones(domain[1]).long().to(device))      
                        loss = ce_loss + domain_loss_t + domain_loss_s 

                    elif args.transfer_method == "mdd":
                        y_s, y_t = output[:domain[0]], output[domain[0]:]
                        y_s_adv, y_t_adv = output_adv[:domain[0]], output_adv[domain[0]:]

                        cls_loss = loss_fn(y_s, label_s) if phase == "train" else loss_fn(logits, label)
 
                        mdd_loss = -mdd_loss_fn(y_s, y_s_adv, y_t, y_t_adv) if phase == "train" else 0
                        loss = cls_loss + mdd_loss * trade_off

                    else:
                        loss = loss_fn(logits, label)
                    metrics['loss'].update(loss.item(), args.batch_size)
                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()
                if args.scheduler in ['one_cycle', 'cyclic']:
                    scheduler.step()
                if args.warmup:
                    warmup_scheduler.dampen()
                evaluation_dict = evaluate_multisource_predictions(label, output, logits=logits, features=features, domain=domain)
                for m, r in evaluation_dict['batch_results'].items():
                    metrics[m].update(r)
                if phase in ['val', 'test']:
                    log_predictions(logger_=logger_predictions,
                                    video_ids=name,
                                    ground_truths=evaluation_dict['ground_truth'],
                                    predictions=evaluation_dict['prediction'])
                    log_logits(logger_=logger_challenge_logits,
                               video_ids=name,
                               logits=evaluation_dict['logits'])
                    log_features(logger_=logger_challenge_features,
                                 video_ids=sample_id,
                                 ground_truths=evaluation_dict['ground_truth'],
                                 features=evaluation_dict['features'])
                metrics['batch_time'].update(time.time() - end_time)
                end_time = time.time()
                if args.display_batch_progress and idx % 10 == 0 and idx != 0:
                    progress.display(idx)
            fig = figure_drawer_log_result(fig, metrics, args.metrics, phase, epoch)
            progress.display(idx)
            log_epoch(sw[phase], log[phase], epoch, metrics)

            if phase == 'val':
                val_metric_loss = metrics['loss'].avg
                val_metric_acc = metrics['acc1'].avg
                if args.scheduler == 'plateu':
                    scheduler.step(metrics=val_metric_loss)
                # validation loss improvement
                if val_metric_loss < best_metric_loss:
                    save_checkpoint(epoch=epoch, metric_suffix='loss',
                                    val_metric=val_metric_loss, best_metric=best_metric_loss,
                                    experiment_path=experiment_folder,
                                    model=model, optimizer=optimizer)
                    best_metric_loss = val_metric_loss
                    is_best = True

                # validation accuracy improvement
                if val_metric_acc > best_metric_acc:
                    save_checkpoint(epoch=epoch, metric_suffix='acc',
                                    val_metric=val_metric_acc, best_metric=best_metric_acc,
                                    experiment_path=experiment_folder,
                                    model=model, optimizer=optimizer)
                    best_metric_acc = val_metric_acc
                    is_best = True

            elif phase == 'test':
                logger.info('==========================================')
            # After each epoch, draw figures
        figure_drawer_draw(fig, args.phase_list, args.metrics)
        if args.scheduler == 'multi_step':
            scheduler.step()
    logger.info('training done.')


def get_training_iteration_count(args, dataloaders):
    iter_len = len(dataloaders['target']) if args.transfer_method == 'single_target' else max(len(dataloaders['train']),
                                                                                              len(dataloaders[
                                                                                                      'target']))
    iter_len = iter_len if args.iterations_per_epoch == -1 else args.iterations_per_epoch
    return iter_len


def initialize_video_dataloaders(args):
    dataloaders = {}
    for ph in args.phase_list:
        if ph == 'train':
            _transform_ph = Compose([
                Resize(size=(args.input_size, args.input_size)),
                RandAugment(),
                ClipToTensor(div_255=True),
                Normalize(mean=args.datasets['bsign22k'].dataset_mean, std=args.datasets['bsign22k'].dataset_std)])
            _dataset_ph = []
            for i in range(len(args.transfer_train_source)):
                _dataset_ph.append(SignLanguageDataset(args=args, split='train', transform=_transform_ph,
                                                       transfer_split=args.transfer_train_source[i]))
            dataloaders['train'] = DataLoader(dataset=torch.utils.data.ConcatDataset(_dataset_ph),
                                              batch_size=args.batch_size // 2, num_workers=args.num_workers,
                                              drop_last=False, shuffle=True)
            _dataset_ph = []
            for i in range(len(args.transfer_train_target)):
                _dataset_ph.append(
                    SignLanguageDataset(args=args, split='train', transform=_transform_ph,
                                        transfer_split=args.transfer_train_target[i]))
            if len(args.transfer_train_target):
                dataloaders['target'] = DataLoader(dataset=torch.utils.data.ConcatDataset(_dataset_ph),
                                                   batch_size=args.batch_size // 2 if args.transfer_method != 'single_target' else args.batch_size,
                                                   num_workers=args.num_workers, drop_last=False, shuffle=True)

        elif ph in ['val', 'test']:
            _transform_ph = Compose([
                Resize(size=(args.input_size, args.input_size)),
                ClipToTensor(div_255=True),
                Normalize(mean=args.datasets['bsign22k'].dataset_mean, std=args.datasets['bsign22k'].dataset_std)
            ])
            _dataset_ph = []
            for i in range(len(args.transfer_validation)):
                _dataset_ph2 = SignLanguageDataset(args=args, split=ph,
                                                   transform=_transform_ph, transfer_split=args.transfer_validation[i])
            dataloaders[ph] = DataLoader(dataset=_dataset_ph2,
                                         batch_size=args.batch_size, num_workers=args.num_workers,
                                         drop_last=False, shuffle=False)
    num_classes = len(_dataset_ph2.unique_labels)
    joint_groups = list(_dataset_ph2.coordinate_joint_groups)
    num_channels = len(joint_groups)

    return num_classes, joint_groups, num_channels, dataloaders

def initialize_coordinate_dataloaders(args, preloaded_data):
    dataloaders = {}
    num_classes = 0
    for ph in args.phase_list:
        if ph == 'train':
            transforms = []
            if args.random_mirror:
                transforms.append(RandomMirror(0.5, args.input_size))
            if args.random_choose:
                transforms.append(RandomChoose(0.5))
            if args.normalization:
                transforms.append(Normalization())
            if args.random_shift:
                transforms.append(RandomShift(0.5))
            _transform_ph = Compose(transforms)


            _dataset_ph = []
            for i in range(len(args.transfer_train_source)):
                _dataset_ph.append(SignLanguageDataset(args=args,
                                          preloaded_data=preloaded_data,
                                          split='train',
                                          transform=_transform_ph,
                                          transfer_split=args.transfer_train_source[i]))
                num_classes = max(len(_dataset_ph[0].unique_labels), num_classes)
            dataloaders['train'] = DataLoader(dataset=torch.utils.data.ConcatDataset(_dataset_ph),
                                              batch_size=args.batch_size // 2, num_workers=args.num_workers,
                                              drop_last=False, shuffle=True)
            _dataset_ph = []
            for i in range(len(args.transfer_train_target)):
                _dataset_ph.append(SignLanguageDataset(args=args,
                                          preloaded_data=preloaded_data,
                                          split='train',
                                          transform=_transform_ph,
                                          transfer_split=args.transfer_train_target[i]))
                num_classes = max(len(_dataset_ph[0].unique_labels), num_classes)
            if len(args.transfer_train_target):
                dataloaders['target'] = DataLoader(dataset=torch.utils.data.ConcatDataset(_dataset_ph),
                                                   batch_size=args.batch_size // 2 if args.transfer_method != 'single_target' else args.batch_size,
                                                   num_workers=args.num_workers, drop_last=False, shuffle=True)

        elif ph in ['val', 'test']:
            transforms = []
            if args.normalization:
                transforms.append(Normalization())
            _transform_ph = Compose(transforms)
            _dataset_ph = []
            for i in range(len(args.transfer_validation)):
                _dataset_ph2 = SignLanguageDataset(args=args,
                                                   preloaded_data=preloaded_data,
                                                   split=ph,
                                                   transform=_transform_ph, transfer_split=args.transfer_validation[i])
            dataloaders[ph] = DataLoader(dataset=_dataset_ph2,
                                         batch_size=args.batch_size, num_workers=args.num_workers,
                                         drop_last=False, shuffle=False)

    num_classes = max(len(_dataset_ph2.unique_labels), num_classes)
    joint_groups = list(_dataset_ph2.coordinate_joint_groups)
    num_channels = len(joint_groups)

    return num_classes, joint_groups, num_channels, dataloaders