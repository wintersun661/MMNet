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
from models import MMNet_original as Model

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


def validation_res(cur_model, epoch_no, opt, logger, target_shape, alpha=0.1):
    resolution = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = opt.batch
    val_dataset = Dataset.CorrespondenceDataset(
        opt.benchmark, opt.data_path, opt.thresh_type, "val", device, opt.resize, opt.max_kps_num)
    val_generator = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch, shuffle=True, num_workers=0)

    cur_model.eval()

    pck_list = []

    for i, data in enumerate(val_generator):
        data["alpha"] = alpha

        cur_batchsize = len(data['src_imname'])
        pred = cur_model(data)

        for k in range(len(data["src_imname"])):

            prd_kps = geometry.predict_kps(
                data["src_kps"][k][:, :data["valid_kps_num"][k]], pred[resolution][0][k], originalShape=target_shape)
            prd_kps = torch.from_numpy(np.array(prd_kps)).to(device)

            pair_pck = evaluation.eval_pck(prd_kps, data, k)
            pck_list.append(pair_pck)

            logger.info('[%5d/%5d]: \t [Pair PCK: %3.3f]\t[Average: %3.3f] %s' %
                        (i*batch_size + k,
                         data['datalen'][0],
                         pair_pck,
                         sum(pck_list) / (i*batch_size+k+1),
                         data['pair_class'][k]))
    res = np.mean(np.array(pck_list))
    logger.info("%d th epoch, PCK res on val set is %.3f" % (epoch_no, res))
    return res


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
        trn_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # model initialization
    model = Model.MMNet(opt).to(device)

    # set loss and optimizer
    criterion = Loss.createLoss(opt)
    optim = Optimizer.createOptimizer(opt, model)

    max_pck = 0.0

    # begin training, iterate over epoch nums
    for epoch in range(epoch_num):

        running_loss = 0.0

        for i, data in enumerate(trn_generator):
            cur_batchsize = len(data['src_imname'])
            pred = model(data)

            loss = 0
            for k in range(len(pred)):
                loss += cross_entropy_loss2d(criterion, pred[k][0], data['src_kps'],
                                             data['trg_kps'], data['valid_kps_num'], target_shape)/cur_batchsize
                loss += cross_entropy_loss2d(criterion, pred[k][1], data['trg_kps'],
                                             data['src_kps'], data['valid_kps_num'], target_shape)/cur_batchsize

            # back propagation
            optim.zero_grad()

            loss.backward()
            optim.step()

            running_loss += loss.item()

            if (i+1) % 20 == 0:
                logger.info("[%d, %5d] loss: %.3f" %
                            (epoch+1, i+1, running_loss/20))
                visualizer.visualize_pred(
                    data, pred, suffix=str(epoch), idx=str(i), visualization_path=os.path.join(ckp_path, opt.visualization_path))

                running_loss = 0.0
        logger.info('saving %dth ckp in %s.' % (epoch, ckp_path))
        torch.save(model.state_dict(), os.path.join(
            ckp_path, str(epoch)+".pth"))
        res = validation_res(model, epoch, opt, logger, target_shape)
        if res > max_pck:
            logger.info('saving %dth ckp as best in %s.' % (epoch, ckp_path))
            torch.save(model.state_dict(), os.path.join(
                ckp_path, "best.pth"))
            max_pck = res


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
