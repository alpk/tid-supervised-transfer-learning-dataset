import pickle
import torch
import json
import time
import glob
import sys
import numpy as np

from pprint import pprint
from utility.logger import *
from matplotlib import pyplot as plt
from utility.meter import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel.data_parallel import DataParallel

def is_windows():
    return sys.platform.startswith('win')


def save_pickle(data, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    with open(save_path + '.pickle', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path + '.pickle', 'rb') as f:
        data = pickle.load(f)
    return data


def read_json(filename, verbose=False):
    with open(os.path.join(filename), 'r') as f:
        data = json.load(f)

    if verbose:
        pprint(data)

    return data

def get_device(verbose=True):
    # use gpu if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if verbose:
        logger.info('Using device, {}'.format(device))
        logger.info('==========================================')
    return device


def initialize_experiment_folder(params):
    param_list = list()
    param_list.append('exp{}'.format(time.strftime("%d.%m.%Y-%H.%M.%S")))
    param_list.append('{}'.format(', '.join(list(params.datasets.keys()))))
    param_list.append('{}'.format(params.transfer_method))
    param_list.append('{}'.format(params.model_type))
    param_list.append('{}'.format(', '.join(map(str, params.bsign_val_user_index))))
    param_list.append('{}'.format(', '.join(map(str,params.transfer_train_source))))
    param_list.append('{}'.format(', '.join(map(str,params.transfer_train_target))))
    param_list.append('{}'.format(', '.join(map(str,params.transfer_validation))))

    param_list.append('c{}'.format(params.clip_length))
    param_list.append('b{}'.format(params.input_size))
    param_list.append('b{}'.format(params.freeze_layer_names))

    param_list.append('b{}'.format(params.batch_size))

    experiment_id = '_'.join([p for p in param_list])
    experiment_path = os.path.join('experiments', experiment_id)
    if not os.path.exists(experiment_path):
        os.makedirs(os.path.join(experiment_path, 'checkpoints'))

    # initialize universal experiment logger
    Logger.__call__().add_file_handler('info', os.path.join(experiment_path, 'experiment.log'))
    Logger.__call__().set_log_level(log_level='info')
    Logger.__call__().add_stream_handler(log_level='info')

    return experiment_path


def write_experiment_config_file(args, checkpoint_dir):
    with open(checkpoint_dir + '/training_args.txt', 'w') as file:
        for a in args.__dict__.keys():
            file.write(a + ': ' +format(args.__dict__[a]) + '\n')


def initialize_epoch_metrics(phase, training_metrics):
    metrics = dict()
    metrics['batch_time'] = AverageMeter('time', ':.3f')
    metrics['data_time'] = AverageMeter('data', ':.3f')
    if phase != 'test':
        metrics['loss'] = AverageMeter('loss', ':.5f')

    for m in training_metrics:
        metrics['{}'.format(m)] = AverageMeter(m, ':.4f')
        metrics['s_' + '{}'.format(m)] = AverageMeter('s_' + m, ':.4f')
        metrics['t_' + '{}'.format(m)] = AverageMeter('t_' + m, ':.4f')

    return metrics


def initialize_loggers(experiment_path, splits, metrics):
    sw = {}
    log = {}
    for s in splits:
        sw_path = os.path.join(experiment_path, 'logs', s)
        if not os.path.exists(sw_path):
            os.makedirs(sw_path)
        sw[s] = SummaryWriter(sw_path)

        mtr_list = ['epoch']
        if s in ['train', 'dev', 'val']:
            mtr_list.append('loss')

        log[s] = CsvLogger(os.path.join(experiment_path, '{}.log'.format(s)), mtr_list + metrics)

    return sw, log


def initialize_figure_drawers(experiment_path, splits, metrics):
    log = {}
    sw_path = os.path.join(experiment_path, 'figures')
    os.makedirs(sw_path, exist_ok=True)
    log['checkpoint_dir'] = sw_path
    for s in splits:
        log[s + '_epoch'] = []
        log[s + '_alignment'] = []
        log[s + '_keyframes'] = []
        for m in metrics:
            log[s + '_' + m] = []
            if s in ['train', 'dev', 'val']:
                log[s + '_loss'] = []
    return log


def figure_drawer_log_result(log, metrics, metric_names, phase, epoch):
    log[phase + '_epoch'].append(epoch)
    for m in metric_names:
        log[phase + '_' + m].append(metrics[m].avg)
    if phase in ['train', 'dev', 'val']:
        log[phase + '_loss'].append(metrics['loss'].avg)
    return log


def figure_drawer_draw(log, phases, metric_names):
    colors = {'train': 'r-', 'dev': 'm-', 'val': 'm-', 'test': 'c-'}
    # Draw Metric Plots
    for m in metric_names:
        for phase in phases:
            plt.plot(np.array(log[phase + '_' + m]), colors[phase])
        plt.legend(phases)
        plt.savefig(os.path.join(log['checkpoint_dir'], m + '.png'))
        plt.clf()

    # Draw Loss Plot
    phases_with_loss = list(set(phases) & set(['train', 'dev', 'val']))

    for phase in phases_with_loss:
        plt.plot(np.array(log[phase + '_loss']), colors[phase])

    if phases_with_loss != []:
        plt.legend(phases_with_loss)
        plt.savefig(os.path.join(log['checkpoint_dir'], 'losses.png'))
        plt.clf()


def log_epoch(sw_, log_, epoch, metrics):
    log_dict = {'epoch': epoch}
    for mtr in log_.header[1:]:
        sw_.add_scalar(mtr, metrics[mtr].avg, epoch)
        log_dict[mtr] = metrics[mtr].avg
    log_.log(log_dict)


def create_epoch_predictions_logger(phase, epoch, experiment_path):
    return CsvLogger(os.path.join(experiment_path,
                                  '{}_predictions'.format(phase),
                                  'epoch_{}.log'.format(epoch)), ['video_id', 'ground_truth', 'prediction'])


def create_epoch_predictions_logger_challenge(phase, epoch, experiment_path):
    return CsvLogger(os.path.join(experiment_path,
                                  '{}_predictions_challenge'.format(phase),
                                  'epoch_{}.csv'.format(epoch)), ['video_id', 'prediction'],
                     delimiter=',')


def create_epoch_logits_logger_challenge(phase, epoch, experiment_path):
    return CsvLogger(os.path.join(experiment_path,
                                  '{}_logits_challenge'.format(phase),
                                  'epoch_{}.csv'.format(epoch)), ['video_id', 'logits'],
                     delimiter=',')

def create_epoch_features_logger_challenge(phase, epoch, experiment_path):
    return CsvLogger(os.path.join(experiment_path,
                                  '{}_features_challenge'.format(phase),
                                  'epoch_{}.csv'.format(epoch)), ['video_id', 'ground_truth', 'features'],
                     delimiter=',')


def log_predictions(logger_, video_ids, predictions, ground_truths=None):
    if ground_truths is not None:
        for pre_idx, pre in enumerate(predictions.T):
            logger_.log({
                'video_id': video_ids[pre_idx],
                'ground_truth': int(ground_truths[pre_idx].cpu().detach().numpy()),
                'prediction': pre.cpu().detach().numpy()
            })
    else:
        for pre_idx, pre in enumerate(predictions.T):
            logger_.log({
                'video_id': video_ids[pre_idx],
                'prediction': pre.cpu().detach().numpy()[0]
            })


def log_logits(logger_, video_ids, logits, ground_truths=None):
    if ground_truths is not None:
        for pre_idx, pre in enumerate(logits):
            logger_.log({
                'video_id': video_ids[pre_idx],
                'ground_truth': int(ground_truths[pre_idx].cpu().detach().numpy()),
                'logits': list(pre.cpu().detach().numpy())
            })
    else:
        for pre_idx, pre in enumerate(logits):
            logger_.log({
                'video_id': video_ids[pre_idx],
                'logits': list(pre.cpu().detach().numpy())
            })


def log_features(logger_, video_ids, features, ground_truths=None):
    if ground_truths is not None:
        for pre_idx, pre in enumerate(features):
            logger_.log({
                'video_id': video_ids[pre_idx],
                'ground_truth': int(ground_truths[pre_idx].cpu().detach().numpy()),
                'features': list(pre.cpu().detach().numpy())
            })
    else:
        for pre_idx, pre in enumerate(features):
            logger_.log({
                'video_id': video_ids[pre_idx],
                'features': list(pre.cpu().detach().numpy())
            })


def evaluate_multilabel_predictions(targets, outputs, topk=(1, 5), logits=None, features=None, multilabel_features=None, multilabel_logits= None):
    acc_list, predictions = accuracy(targets, outputs, topk=topk)
    reg_prediction, mse, score = regression_score(data_y=multilabel_features, pred_y= multilabel_logits, pct_close=0.1)
    acc_dict = {
            'acc1': acc_list[0].cpu().detach().numpy()[0],
            'acc5': acc_list[1].cpu().detach().numpy()[0],
            'mse': float(mse.cpu().detach())
        }

    return {'ground_truth': targets,
            'prediction': predictions,
            'batch_results': acc_dict,
            'logits': logits,
            'features':features,
            'reg_prediction':reg_prediction.cpu().detach()}

def evaluate_multisource_predictions(targets, outputs, topk=(1, 5), logits=None, features=None, domain=None):

    acc_list, predictions = accuracy(targets, outputs, topk=topk)
    if domain[0] > 0:
        acc_list_s, _ = accuracy(targets[:domain[0]], outputs[:domain[0],:], topk=topk)
        acc_list_t, _ = accuracy(targets[domain[0]:], outputs[domain[0]:, :], topk=topk)
    acc_dict = {
            'acc1': acc_list[0].cpu().detach().numpy()[0],
            'acc5': acc_list[1].cpu().detach().numpy()[0],
            's_acc1': acc_list_s[0].cpu().detach().numpy()[0] if domain[0] > 0 else -1,
            's_acc5': acc_list_s[1].cpu().detach().numpy()[0] if domain[0] > 0 else -1,
            't_acc1': acc_list_t[0].cpu().detach().numpy()[0] if domain[0] > 0 else -1,
            't_acc5': acc_list_t[1].cpu().detach().numpy()[0] if domain[0] > 0 else -1,
        }

    return {'ground_truth': targets,
            'prediction': predictions,
            'batch_results': acc_dict,
            'logits': logits,
            'features':features}


def save_checkpoint(epoch, val_metric, best_metric, experiment_path, model, optimizer, metric_suffix='loss'):
    save_file = 'model_epoch{}_{}{:.4f}.pth'.format(epoch, metric_suffix, val_metric)
    logger.info('Validation [{}] is improved from {:.4f} to {:.4f}, saving model to {}'
                .format(metric_suffix, best_metric, val_metric, save_file))

    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               os.path.join(experiment_path, 'checkpoints', save_file))
    prev_save_file = os.path.join(experiment_path, 'checkpoints', save_file)

    return prev_save_file


def accuracy(target, output, topk=(1,)):
    """
    https://github.com/bearpaw/pytorch-classification/blob/cc9106d598ff1fe375cc030873ceacfea0499d77/utils/eval.py
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res, pred

def regression_score(data_y ,pred_y , pct_close=0.15):
    n_items = data_y.shape[0]

    n_wrong = 0
    with torch.no_grad():
        correct_mat = (torch.abs(pred_y - data_y) < pct_close).short()
        lse = torch.mean(torch.square((pred_y - data_y).float()))
        n_correct = torch.sum(correct_mat,dim=1)
        n_wrong = torch.tensor([data_y.shape[1] for x in range(n_items)]).cuda() - n_correct
    return pred_y, lse, n_correct / (n_correct + n_wrong)



def freeze_layers(epoch, model, name_to_freeze, epochs_to_freeze, freeze_all_bn=True, verbose=True):
    _bn_funcs = (
        torch.nn.modules.BatchNorm1d,
        torch.nn.modules.BatchNorm2d,
        torch.nn.modules.BatchNorm3d,
    )

    if epoch in epochs_to_freeze:
        ct = []
        name_to_freeze = name_to_freeze[epochs_to_freeze.index(epoch)]

        if isinstance(model, DataParallel):
            for name, child in model.module.named_children():
                if name_to_freeze in ct or name_to_freeze == '':
                    for params in child.parameters():
                        params.requires_grad = True
                else:
                    for params in child.parameters():
                        params.requires_grad = False
                ct.append(name)
        else:
            for name, child in model.named_children():
                if name_to_freeze in ct or name_to_freeze == '':
                    for params in child.parameters():
                        params.requires_grad = True
                else:
                    for params in child.parameters():
                        params.requires_grad = False
                ct.append(name)

        # Freeze All Batchnorm Layers
        if isinstance(model, DataParallel):
            if freeze_all_bn:
                for module in model.module.modules():
                    if isinstance(module, _bn_funcs):
                        try:
                            module.weight.requires_grad = False
                            module.bias.requires_grad = False
                        except:
                            module.running_mean.requires_grad = False
                            module.running_var.requires_grad = False
        else:
            if freeze_all_bn:
                for module in model.modules():
                    if isinstance(module, _bn_funcs):
                        try:
                            module.weight.requires_grad = False
                            module.bias.requires_grad = False
                        except:
                            module.running_mean.requires_grad = False
                            module.running_var.requires_grad = False

        # To view which layers are freeze and which layers are not freezed:
        if verbose:
            if isinstance(model, DataParallel):
                for name, child in model.module.named_children():
                    buf = ''
                    for name_2, params in child.named_parameters():
                        buf = buf + name + ' ' + name_2 + ':' + str(params.requires_grad) + ', '
                    logger.info(buf)
            else:
                for name, child in model.named_children():
                    buf = ''
                    for name_2, params in child.named_parameters():
                        buf = buf + name + ' ' + name_2 + ':' + str(params.requires_grad) + ', '
                    logger.info(buf)

    return model