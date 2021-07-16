import torch
import numpy as np
import os
import random
from torch._C import device
import torch.nn.functional as F

# utility import
from logger import BaseLogger as Logger
from options import TrainOptions as Options

# data processing
from data import PascalDataset as Dataset
#from utils import visualizer

from models import Loss, Optimizer
from models import Model as Model

from utils import geometry, visualizer
from evaluation_tools import evaluation


def fix_random_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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


def adjust_learning_rate(optimizer, gamma=0.1, logger=None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma
        if logger:
            logger.info('%s: %s' % (param_group['name'], param_group['lr']))


def train(logger, opt):
    # insanity check
    if not os.path.isdir(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)

    ckp_path = opt.checkpoint_path
    batch_size = opt.batch
    epoch_num = opt.epoch

    height, width = opt.resize.split(',')
    target_shape = [int(width), int(height)]

    # set specified gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set up random seed
    fix_random_seed()

    #  data input
    trn_dataset = Dataset.CorrespondenceDataset(
        opt.benchmark, opt.data_path, opt.thresh_type, "trn", device, opt.resize, opt.max_kps_num)
    trn_generator = torch.utils.data.DataLoader(
        trn_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    trn_size = len(trn_generator)
    data_iter = iter(trn_generator)

    # model initialization
    model = Model.MMNet(opt).to(device)

    # set loss and optimizer
    criterion = Loss.createLoss(opt)
    optim = Optimizer.createOptimizer(opt, dict(model.named_parameters()))
    # model.backbone.train()
    model.train()
    max_iter = opt.epoch * len(trn_generator)
    epoch_loss = 0

    step = 1
    cur = 0
    max_pck = 0.0
    running_loss = 0.0
    # begin training, iterate over epoch nums
    for step in range(1, max_iter + 1):
        optim.zero_grad()
        epoch = step / len(trn_generator)
        data = next(data_iter)
        cur_batchsize = len(data['src_imname'])
        pred = model(data)
        loss = 0

        for k in range(len(pred)):
            loss += cross_entropy_loss2d(criterion, pred[k][0], data['src_kps'],
                                         data['trg_kps'], data['valid_kps_num'], target_shape)/cur_batchsize
            loss += cross_entropy_loss2d(criterion, pred[k][1], data['trg_kps'],
                                         data['src_kps'], data['valid_kps_num'], target_shape)/cur_batchsize

        # back propagation
        loss.backward()
        optim.step()

        running_loss += loss.item()

        cur += 1
        if step % opt.step_size == 0:
            Optimizer.adjust_learning_rate(optim, opt.gamma, logger)

        if step % opt.log_interval == 0:
            logger.info("[%d, %5d] loss: %.3f" %
                        (epoch+1, cur+1, running_loss/50))
            visualizer.visualize_pred(
                data, pred, suffix=str(int(epoch)), idx=str(cur), visualization_path=os.path.join(ckp_path, opt.visualization_path))

            running_loss = 0.0
        if cur == len(trn_generator):
            cur = 0
            # valid
            data_iter = iter(trn_generator)
            logger.info('saving %dth ckp in %s.' % (epoch, ckp_path))
            torch.save(model.state_dict(), os.path.join(
                ckp_path, str(epoch)+".pth"))
            # evaluate pck with specified settings
            val_resolution = opt.val_resolution

            val_dataset = Dataset.CorrespondenceDataset(
                opt.benchmark, opt.data_path, opt.thresh_type, "val", device, opt.resize, opt.max_kps_num)
            val_generator = torch.utils.data.DataLoader(
                val_dataset, batch_size=opt.val_batch, shuffle=True, num_workers=0)

            # model.eval()
            model.backbone.eval()

            pck_list = []

            for i, data in enumerate(val_generator):
                data["alpha"] = opt.val_alpha
                with torch.no_grad():
                    cur_batchsize = len(data['src_imname'])
                    pred = model(data)

                    for k in range(cur_batchsize):

                        prd_kps = geometry.predict_kps(
                            data["src_kps"][k][:, :data["valid_kps_num"][k]], pred[val_resolution][0][k], originalShape=target_shape)
                        prd_kps = torch.from_numpy(
                            np.array(prd_kps)).to(device)

                        pair_pck = evaluation.eval_pck(prd_kps, data, k)
                        pck_list.append(pair_pck)

                        # logger.info('[%5d/%5d]: \t [Pair PCK: %3.3f]\t[Average: %3.3f] %s' %
                        #             (i*opt.val_batch + k+1,
                        #              data['datalen'][0],
                        #              pair_pck,
                        #              sum(pck_list) / (i*batch_size+k+1),
                        #              data['pair_class'][k]))

            res = np.mean(np.array(pck_list))
            logger.info("%d th epoch, PCK res on val set is %.3f" %
                        (epoch, res))
            if res > max_pck:
                logger.info('saving %dth ckp as best in %s.' %
                            (epoch, ckp_path))
                torch.save(model.state_dict(), os.path.join(
                    ckp_path, "best.pth"))
                max_pck = res

            model.backbone.train()
            model.train()

    logger.info('Training completed. Best result on val with alpha %.2f at resolution %d is %.3f.' % (
        opt.val_alpha, opt.val_resolution, max_pck))


if __name__ == "__main__":
    print("currently executing train.py file.")

    options = Options.OptionParser().parse()
    logger = Logger.Logger(file_path=options.checkpoint_path,
                           time_stamp=True, suffix="train").createLogger()

    args = vars(options)
    logger.info("Options listed below:----------------")
    for k, v in args.items():
        logger.info("%s: %s" % (str(k), str(v)))
    logger.info("Options all listed.------------------")
    train(logger, options)
