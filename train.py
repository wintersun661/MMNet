import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
import argparse
import time

from logger.BaseLogger import Logger
import re
import random
import os
import sys
import math
from models import Model as Model
from utils import visualizer, geometry
from evaluation_tools import evaluation

from options import TrainOptions as Options
import datetime
import cv2


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, gamma=0.1, logger=None):
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


def train(model, args, logger):
    datapath = args.data_path
    benchmark = args.benchmark

    if benchmark == 'pfpascal':
        from data import PascalDataset as Dataset
    if benchmark == 'spair':
        from data import SpairDataset as Dataset

    thres = args.thresh_type
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_interval = args.log_interval
    best_avePCK = 0.0

    height, width = args.resize.split(',')
    target_shape = [int(width), int(height)]
    # loss_func = nn.CrossEntropyLoss(reduction='mean')

    loss_func = nn.BCELoss(size_average=False)

    # trainning data benchmark initialization

    trainDset = Dataset.CorrespondenceDataset(
        benchmark, datapath, thres, 'trn', device, args.resize, args.max_kps_num)

    trainDataloader = DataLoader(
        trainDset, batch_size=args.batch, num_workers=0, shuffle=True)
    valDset = Dataset.CorrespondenceDataset(
        benchmark, datapath, thres, 'val', device, args.resize, args.max_kps_num)
    valDataloader = DataLoader(
        valDset, batch_size=args.val_batch, num_workers=0)

    params_dict = dict(model.named_parameters())
    base_lr = args.lr
    weight_decay = args.weight_decay

    alpha = args.val_alpha
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
                                lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(params, lr=0.001, betas=(
    #     0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

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

    model.train()

    max_iter = args.epoch * iter_per_epoch
    epoch_loss = 0

    for step in range(start_step, max_iter + 1):
        optimizer.zero_grad()
        batch_loss = 0
        epoch = step/len(trainDataloader)

        data = next(data_iter)

        out = model(data)
        curBatchSize = len(data['src_img'])

        loss = 0

        for k in range(4):

            loss += cross_entropy_loss2d(loss_func, out[k][0], data['src_kps'],
                                         data['trg_kps'], data['valid_kps_num'], originalShape=target_shape)/curBatchSize
            loss += cross_entropy_loss2d(loss_func, out[k][1], data['trg_kps'],
                                         data['src_kps'], data['valid_kps_num'], originalShape=target_shape)/curBatchSize
            # print(loss)

        torch.autograd.set_detect_anomaly(True)
        loss.backward()

        batch_loss += loss.cpu().data.numpy()
        epoch_loss += batch_loss
        if cur % log_interval == 0:
            visualizer.visualize_pred(
                data, out, suffix=str(int(epoch)), idx=str(cur), visualization_path=os.path.join(args.ckp_path, 'train_vis'))

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
            adjust_learning_rate(optimizer, args.gamma, logger)

        if step % 300 == 0:
            tm = time.time() - start_time
            logger.info('iter: %d, lr: %e, loss: %f, time using: %f(%fs/iter)' % (step,
                                                                                  optimizer.param_groups[0]['lr'], np.mean(mean_loss), tm, tm/300))
            start_time = time.time()

        if cur == iter_per_epoch:
            cur = 0
            epoch_loss /= len(trainDataloader)
            logger.info('TRAIN set: Average loss: {:.4f}'.format(epoch_loss))
            epoch_loss = 0
            torch.save(model.state_dict(), os.path.join(
                args.ckp_path, str(int(step/iter_per_epoch))+'.pth'))
            model.eval()
            model.backbone.eval()
            PCK_list = []
            for idx, data in enumerate(valDataloader):

                data['alpha'] = args.val_alpha
                # epoch_loss = 0.
                with torch.no_grad():

                    out = model(data)

                    curBatchSize = len(data['src_img'])

                    for i in range(len(data['src_kps'])):

                        prd_kps = geometry.predict_kps(
                            data['src_kps'][i][:, :data['valid_kps_num'][i]], out[args.val_resolution][0][i], originalShape=target_shape)
                        prd_kps = torch.from_numpy(
                            np.array(prd_kps)).to(device)

                        pair_pck = evaluation.eval_pck(prd_kps, data, i)
                        PCK_list.append(pair_pck)

            avePCK = sum(PCK_list)/len(PCK_list)
            logger.info('VAL set: Average pck: {:.4f}'.format(avePCK))
            is_best = avePCK > best_avePCK
            best_avePCK = max(avePCK, best_avePCK)
            if is_best:
                torch.save(model.state_dict(),
                           os.path.join(args.ckp_path, 'best.pth'))
            model.backbone.train()
            model.train()

            data_iter = iter(trainDataloader)
    logger.info('Training completed. Best result on val with alpha %.2f at resolution %d is %.3f.' % (
        args.val_alpha, args.val_resolution, best_avePCK))


def main():
    args = Options.OptionParser().parse()

    logger = Logger(file_path=args.ckp_path,
                    time_stamp=True, suffix="train").createLogger()
    logger.info('*'*80)
    logger.info('the args are the below')
    logger.info('*'*80)
    for x in args.__dict__:
        logger.info(x+','+str(args.__dict__[x]))
    logger.info(args.benchmark)
    logger.info('*'*80)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda:0'

    setup_seed(8000)
    model = Model.MMNet(args).to(device)

    logger.info(model)
    train(model, args, logger)


if __name__ == '__main__':
    main()
