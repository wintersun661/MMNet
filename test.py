import time
import torch
import numpy as np
import os
import random
import torch.nn.functional as F

# utility import
from logger import BaseLogger as Logger
from options import TestOptions as Options

# data processing
from data import PascalDataset as Dataset
# from utils import visualizer

from models import Loss, Optimizer
from models import Model as Model

from utils import geometry
from evaluation_tools import evaluation


def fix_random_seed(seed=121):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def test(logger, opt):

    # insanity check for ckp file
    if not os.path.isdir(opt.checkpoint_path):
        logger.error("Null checkpoint file list!")
        exit()

    ckp_path = opt.checkpoint_path
    ckp_name = opt.ckp_type+'.pth'

    if opt.ckp_type == 'latest':
        ckp_name = str(opt.epoch)+'.pth'

    #ckp_fullname = os.path.isfile(os.path.join(ckp_path, ckp_name))
    ckp_fullname = "/home/zysong/SC_BDCN/ckp_fcn21_nonlocal/2021-03-09_08:25_best"

    if not os.path.isfile(ckp_fullname):
        logger.error("Null checkpoint file!")
        exit()

    batch_size = opt.batch

    height, width = opt.resize.split(',')
    target_shape = [int(width), int(height)]

    alpha = opt.alpha

    # set specified gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set up random seed
    fix_random_seed()

    #  data input
    test_dataset = Dataset.CorrespondenceDataset(
        opt.benchmark, opt.data_path, opt.thresh_type, "test", device, opt.resize, opt.max_kps_num)
    test_generator = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # model initialization
    model = Model.MMNet(opt).to(device)
    model.load_state_dict(torch.load(ckp_fullname))

    zero_pcks = 0
    time_list = []
    pck_list = []

    # iterate over test set
    for i, data in enumerate(test_generator):
        data["alpha"] = alpha

        cur_batchsize = len(data['src_imname'])
        pred = model(data)

        for k in range(len(data["src_imname"])):
            tic = time.time()
            prd_kps = geometry.predict_kps(
                data["src_kps"][k][:, :data["valid_kps_num"][k]], pred[opt.resolution][0][k], originalShape=target_shape)
            prd_kps = torch.from_numpy(np.array(prd_kps)).to(device)
            toc = time.time()

            pair_pck = evaluation.eval_pck(prd_kps, data, k)
            pck_list.append(pair_pck)

            logger.info('[%5d/%5d]: \t [Pair PCK: %3.3f]\t[Average: %3.3f] %s' %
                        (i*batch_size + k,
                         data['datalen'][0],
                         pair_pck,
                         sum(pck_list) / (i*batch_size+k+1),
                         data['pair_class'][k]))


if __name__ == "__main__":
    print("currently executing test.py file.")

    logger = Logger.Logger(suffix='test', time_stamp=True).createLogger()

    options = Options.OptionParser().parse()

    args = vars(options)
    logger.info("Options listed below:----------------")
    for k, v in args.items():
        logger.info("%s: %s" % (str(k), str(v)))
    logger.info("Options all listed.------------------")
    test(logger, options)