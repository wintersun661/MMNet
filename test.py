import time
import torch
import numpy as np
import os
import random
import torch.nn.functional as F

from utils import visualizer

# utility import
from logger import BaseLogger as Logger
from options import TestOptions as Options

from models import Model as Model

from utils import geometry as geometry
from evaluation_tools import evaluation


def fix_random_seed(seed=121):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def test(logger, opt):

    # insanity check for ckp file
    # if not os.path.isdir(opt.checkpoint_path):
    #     logger.error("Null checkpoint file list!")
    #     exit()


    ckp_fullname = opt.ckp_name

    logger.info('ckp file: % s' % ckp_fullname)
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

    benchmark = opt.benchmark

    if benchmark == 'pfpascal':
        from data import PascalDataset as Dataset
    if benchmark == 'spair':
        from data import SpairDataset as Dataset

    # set up random seed
    fix_random_seed()

    #  data input
    test_dataset = Dataset.CorrespondenceDataset(
        opt.benchmark, opt.data_path, opt.thresh_type, "test", device, opt.resize, opt.max_kps_num)
    test_generator = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # model initialization
    model = Model.MMNet(opt).to(device)
    model.backbone.eval()
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
            # visualizer.vis_corr(data['src_img'][k], data['trg_img'][k], data["src_kps"][k]
            #                     [:, :data["valid_kps_num"][k]], data["trg_kps"][k][:, :data["valid_kps_num"][k]], prd_kps,
            #                     'visualized_test_img', i*batch_size+k)
            pair_pck = evaluation.eval_pck(prd_kps, data, k)
            pck_list.append(pair_pck)

            logger.info('[%5d/%5d]: \t [Pair PCK: %3.3f]\t[Average: %3.3f] %s' %
                        (i*batch_size + k,
                         data['datalen'][0],
                         pair_pck,
                         sum(pck_list) / (i*batch_size+k+1),
                         data['pair_class'][k]))
        # if i % 50 == 0:
        #     visualizer.visualize_pred(
        #         data, pred, suffix="", idx=str(i), visualization_path=os.path.join(opt.log_path, opt.visualization_path, 'test'))


if __name__ == "__main__":
    print("currently executing test.py file.")

    options = Options.OptionParser().parse()
    logger = Logger.Logger(file_path=options.log_path,
                           time_stamp=True, suffix="test").createLogger()
    args = vars(options)
    logger.info("Options listed below:----------------")
    for k, v in args.items():
        logger.info("%s: %s" % (str(k), str(v)))
    logger.info("Options all listed.------------------")
    test(logger, options)
