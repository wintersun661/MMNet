import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
import argparse
import time

import re
import random
import os
import sys
import math
from models import MMNet_original as Model
from utils import visualizer, geometry
from data import PascalDataset as Dataset
import datetime
from logger import BaseLogger as Logger

import logging as log
import cv2


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子


def adjust_learning_rate(optimizer, steps, step_size, gamma=0.1, logger=None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma
        if logger:
            logger.info('%s: %s' % (param_group['name'], param_group['lr']))


def buildOneHot(index, featsShape):
    res = torch.zeros(featsShape[0]*featsShape[1])
    res[index] = 1
    return res


def cross_entropy_loss2d(loss_func, inputs, kps_src_list, kps_trg_list, effect_num_list, originalShape, cuda=True):
    # loss
    """
    :param inputs: inputs is a 5 dimensional data nx(h,w)x(h,W)
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    b, h, w = inputs.shape[:3]

    featsShape = [h, w]

    loss = 0

    for i in range(b):
        kps_src = kps_src_list[i].cpu().data.numpy()
        kps_trg = kps_trg_list[i].cpu().data.numpy()
        weights = torch.zeros(effect_num_list[i], h*w)
        targets = torch.zeros(effect_num_list[i], h*w)

        for j in range(0, effect_num_list[i]):

            weights[j] = geometry.BilinearInterpolate(
                [kps_src[0, j], kps_src[1, j]], inputs[i], originalShape)
            targets[j] = geometry.getBlurredGT(
                [kps_trg[0, j], kps_trg[1, j]], featsShape, originalShape)

        # print(loss_func(F.softmax(weights),targets))
        loss += loss_func(F.softmax(weights), targets)

    return loss


def cross_entropy_loss2d_onehot(loss_func, inputs, kps_src_list, kps_trg_list, effect_num_list, cuda=True):

    # loss
    """
    :param inputs: inputs is a 5 dimensional data nx(h,w)x(h,W)
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    b, h, w = inputs.shape[:3]

    featsShape = [h, w]

    loss = 0

    for i in range(b):
        kps_src = kps_src_list[i].cpu().data.numpy()
        kps_trg = kps_trg_list[i].cpu().data.numpy()
        weights = torch.zeros(effect_num_list[i], h*w)
        targets = torch.zeros(effect_num_list[i], dtype=torch.int64)

        for j in range(0, effect_num_list[i]):

            x, y = geometry.findNearestPoint(
                [kps_src[0, j], kps_src[1, j]], featsShape)

            weights[j, :] = inputs[i, int(y), int(x), :].view(-1)

            x, y = geometry.findNearestPoint(
                [kps_trg[0, j], kps_trg[1, j]], featsShape)
            index = int(y * w + x)

            targets[j] = index

        loss += loss_func(weights, targets)

    return loss


def train(model, args):
    datapath = args.datapath
    benchmark = args.dataset
    height, width = args.resize.split(',')
    target_shape = [int(width), int(height)]
    ckp_path = args.ckp_path
    thres = args.thres
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_interval = args.log_interval
    best_val_loss = float("inf")
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    # loss_func = nn.CrossEntropyLoss(reduction='mean')
    loss_func = nn.BCELoss(size_average=False)

    # trainning data benchmark initialization
    setup_seed(1)

    trainDset = Dataset.CorrespondenceDataset(
        'pfpascal', args.datapath, args.thres, "trn", device, args.resize, 50)
    trainDataloader = DataLoader(
        trainDset, batch_size=5, num_workers=0, shuffle=True)
    valDset = Dataset.CorrespondenceDataset(
        'pfpascal', args.datapath, args.thres, "val", device, args.resize, 50)
    valDataloader = DataLoader(
        valDset, batch_size=5, num_workers=0)

    params_dict = dict(model.named_parameters())
    base_lr = args.base_lr
    weight_decay = args.weight_decay
    logger = args.logger
    params = []

    for key, v in params_dict.items():

        if re.match(r'conv[1-5]_[1-9]*_down', key):
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr*0.1,
                            'weight_decay': weight_decay*1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr*0.2,
                            'weight_decay': weight_decay*0, 'name': key}]
        elif re.match(r'conv[1-5]_kernel', key):
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr*1,
                            'weight_decay': weight_decay*1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr*2,
                            'weight_decay': weight_decay*0, 'name': key}]
        elif re.match(r'conv[1-5]_scale ', key):
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr*1,
                            'weight_decay': weight_decay*1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr*2,
                            'weight_decay': weight_decay*0, 'name': key}]
        elif re.match(r'.*upsample_[1-9]*', key):

            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr,
                            'weight_decay': weight_decay, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr,
                            'weight_decay': weight_decay*0, 'name': key}]
        elif re.match(r'msblock[1-5]_[1-9]*\.conv', key):

            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr*1,
                            'weight_decay': weight_decay*1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr*2,
                            'weight_decay': weight_decay*0, 'name': key}]
        elif re.match(r'fuse', key):
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr*1,
                            'weight_decay': weight_decay*1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr*2,
                            'weight_decay': weight_decay*0, 'name': key}]
        else:
            if 'weight' in key:
                params += [{'params': v, 'lr': base_lr*0.001,
                            'weight_decay': weight_decay*1, 'name': key}]
            elif 'bias' in key:
                params += [{'params': v, 'lr': base_lr*0.002,
                            'weight_decay': weight_decay*0, 'name': key}]

    optimizer = torch.optim.SGD(params, momentum=args.momentum,
                                lr=args.base_lr, weight_decay=args.weight_decay)

    start_step = 1
    mean_loss = []
    cur = 0
    pos = 0
    data_iter = iter(trainDataloader)
    iter_per_epoch = len(trainDataloader)
    logger.info('*'*40)
    logger.info('train images in all are %d ' % iter_per_epoch)
    logger.info('*'*40)
    for param_group in optimizer.param_groups:
        if logger:
            logger.info('%s: %s' % (param_group['name'], param_group['lr']))
    start_time = time.time()
    if args.cuda:
        model.cuda()
    if args.resume:
        logger.info('resume from %s' % args.resume)
        state = torch.load(args.resume)
        start_step = state['step']
        optimizer.load_state_dict(state['solver'])
        model.load_state_dict(state['param'])
    model.train()
    batch_size = args.batch_size
    max_iter = args.epoch * iter_per_epoch
    epoch_loss = 0

    for step in range(start_step, max_iter + 1):
        optimizer.zero_grad()
        batch_loss = 0
        for i in range(args.iter_size):
            if cur == iter_per_epoch:
                cur = 0
                epoch_loss /= len(trainDataloader)
                print('TRAIN set: Average loss: {:.4f}'.format(epoch_loss))
                epoch_loss = 0
                if not os.path.isdir(ckp_path):
                    os.mkdirs(ckp_path)
                torch.save(model.state_dict(), os.path.join(ckp_path, cur_time +
                                                            '_'+str(int(step/iter_per_epoch))+'.pth.tar'))
                for idx, data in enumerate(valDataloader):
                    tic_val = time.time()
                    # epoch_loss = 0.
                    with torch.no_grad():

                        out = model(data)

                        curBatchSize = len(data['src_img'])

                        loss = 0
                        for k in range(4):

                            loss += cross_entropy_loss2d(
                                loss_func, out[k][0], data['src_kps'], data['trg_kps'], data['valid_kps_num'], target_shape)/curBatchSize
                            loss += cross_entropy_loss2d(
                                loss_func, out[k][1], data['trg_kps'], data['src_kps'], data['valid_kps_num'], target_shape)/curBatchSize

                        loss_np = loss.data.cpu().numpy()

                        epoch_loss += loss_np

                epoch_loss /= len(valDataloader)
                print('VAL set: Average loss: {:.4f}'.format(epoch_loss))
                is_best = epoch_loss < best_val_loss
                best_val_loss = min(epoch_loss, best_val_loss)
                if is_best:
                    torch.save(model.state_dict(),
                               os.path.join(ckp_path, cur_time+'_best'))

                data_iter = iter(trainDataloader)
            data = next(data_iter)
            if args.cuda:
                data = data.cuda()
            out = model(data)
            curBatchSize = len(data['src_img'])

            loss = 0
            for k in range(4):
                loss += cross_entropy_loss2d(loss_func, out[k][0], data['src_kps'],
                                             data['trg_kps'], data['valid_kps_num'], target_shape)/curBatchSize
                loss += cross_entropy_loss2d(loss_func, out[k][1], data['trg_kps'],
                                             data['src_kps'], data['valid_kps_num'], target_shape)/curBatchSize

            loss.backward()
            # visualizer.visualizer_in_one(data,cur,out)
            '''
            for name, parms in model.named_parameters():
                print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
                ' -->grad_value:',parms.grad, ' -->weight:', parms.data)
            '''
            batch_loss += loss.cpu().data.numpy()
            epoch_loss += batch_loss
            if cur % log_interval == 0:
                visualizer.visualize_pred(
                    data, out, suffix=str(int(step/iter_per_epoch)), idx=str(i), visualization_path=os.path.join(ckp_path, args.visualizer))

                logger.info('TRAIN Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(int(
                    step/iter_per_epoch), cur, iter_per_epoch, 100. * cur / len(trainDataloader), batch_loss))
            cur += 1
        # update parameter
        optimizer.step()
        if len(mean_loss) < args.average_loss:
            mean_loss.append(batch_loss)
        else:
            mean_loss[pos] = batch_loss
            pos = (pos + 1) % args.average_loss
        if step % args.step_size == 0:
            adjust_learning_rate(optimizer, step, args.step_size, args.gamma)
        if step % args.snapshots == 0:
            torch.save(model.state_dict(), '%s/bdcn_%d.pth' %
                       (args.param_dir, step))
            state = {'step': step+1, 'param': model.state_dict(),
                     'solver': optimizer.state_dict()}
            torch.save(state, '%s/bdcn_%d.pth.tar' % (args.param_dir, step))
        if step % args.display == 0:
            tm = time.time() - start_time
            logger.info('iter: %d, lr: %e, loss: %f, time using: %f(%fs/iter)' % (step,
                                                                                  optimizer.param_groups[0]['lr'], np.mean(mean_loss), tm, tm/args.display))
            start_time = time.time()


def main():
    args = parse_args()

    logger = Logger.Logger(file_path=args.ckp_path,
                           time_stamp=True, suffix="train").createLogger()

    args.logger = logger

    logger.info('*'*80)
    logger.info('the args are the below')
    logger.info('*'*80)
    for x in args.__dict__:
        logger.info(x+','+str(args.__dict__[x]))
    logger.info(args.dataset)
    logger.info('*'*80)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda:0'
    if not os.path.exists(args.param_dir):
        os.mkdir(args.param_dir)
    torch.manual_seed(time.time())

    model = Model.MMNet().to(device)
    if args.complete_pretrain:
        model.load_state_dict(torch.load(args.complete_pretrain))
    logger.info(model)
    train(model, args)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train BDCN for different args')
    parser.add_argument('--datapath', type=str,
                        default='../MMNet_Feb_15/Datasets_SCOT')
    parser.add_argument('-d', '--dataset', type=str,
                        default='pfpascal', help='The dataset to train')
    parser.add_argument('--param-dir', type=str, default='params',
                        help='the directory to store the params')
    parser.add_argument('--lr', dest='base_lr', type=float, default=1e-6,
                        help='the base learning rate of model')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='the momentum')
    parser.add_argument('-c', '--cuda', action='store_true',
                        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='the gpu id to train net')
    parser.add_argument('--weight-decay', type=float, default=0.0002,
                        help='the weight_decay of net')
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='whether resume from some, default is None')
    parser.add_argument('-p', '--pretrain', type=str, default=None,
                        help='init net from pretrained model default is None')
    parser.add_argument('--max-iter', type=int, default=40000,
                        help='max iters to train network, default is 40000')
    parser.add_argument('--epoch', type=int, default=20,
                        help='epochs to train network, default is 20')
    parser.add_argument('--iter-size', type=int, default=1,
                        help='iter size for back propagation, default 1')
    parser.add_argument('--average-loss', type=int, default=50,
                        help='smoothed loss, default is 50')
    parser.add_argument('-s', '--snapshots', type=int, default=1000,
                        help='how many iters to store the params, default is 1000')
    parser.add_argument('--step-size', type=int, default=10000,
                        help='the number of iters to decrease the learning rate, default is 10000')
    parser.add_argument('--display', type=int, default=20,
                        help='how many iters display one time, default is 20')
    parser.add_argument('-b', '--balance', type=float, default=1.1,
                        help='the parameter to balance the neg and pos, default is 1.1')
    parser.add_argument('-l', '--log', type=str, default='log.txt',
                        help='the file to store log, default is log.txt')
    parser.add_argument('-k', type=int, default=1,
                        help='the k-th split set of multicue')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='batch size of one iteration, default 5')
    parser.add_argument('--crop-size', type=int, default=None,
                        help='the size of image to crop, default not crop')
    parser.add_argument('--yita', type=float, default=None,
                        help='the param to operate gt, default is data in the config file')
    parser.add_argument('--complete-pretrain', type=str, default=None,
                        help='finetune on the complete_pretrain, default None')
    parser.add_argument('--side-weight', type=float, default=0.5,
                        help='the loss weight of sideout, default 0.5')
    parser.add_argument('--fuse-weight', type=float, default=1.1,
                        help='the loss weight of fuse, default 1.1')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='the decay of learning rate, default 0.1')
    parser.add_argument('--thres', type=str, default='auto',
                        choices=['auto', 'img', 'bbox'])
    parser.add_argument('--log_interval', type=float, default=10,
                        help='the log interval,default = 10')
    parser.add_argument('--visualizer', type=str,
                        default='visualized_training_')
    parser.add_argument('--ckp_path', type=str, default="ckp_train_feb_15")
    parser.add_argument('--resize', type=str, default="224,320")
    return parser.parse_args()


if __name__ == '__main__':
    main()
