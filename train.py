import torch
import os

# utility import
from logger import BaseLogger as Logger
from options import TrainOptions as Options

# data processing
from data import BaseDataset as Dataset
#from utils import visualizer

from models import Loss, Optimizer
from models import Model as Model


def fix_random_seed(seed=121):
    torch.manual_seed(seed)


def train(logger, opt):
    # insanity check
    if not os.path.isdir(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)

    ckp_path = opt.checkpoint_path
    batch_size = opt.batch
    epoch_num = opt.epoch
    lr = opt.lr
    momentum = opt.momentum

    # set specified gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #  data input
    trn_dataset = Dataset.CustomizedDataset(filePath=opt.train_path)
    trn_generator = torch.utils.data.Dataloader(
        trn_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # model initialization
    model = Model.Model().to(device)

    # set loss and optimizer
    criterion = Loss.createLoss(opt)
    optim = Optimizer.createOptimizer(opt, model)

    # begin training, iterate over epoch nums
    for epoch in range(epoch_num):
        running_loss = 0.0

        for i, data in enumerate(trn_generator):
            X, y = data
            X = X.to(device)
            y = y.to(device)
            pred = model(X)

            # back propagation
            optim.zero_grad()
            loss = criterion(pred, y)
            loss.backward()
            optim.step()

            running_loss += loss.item()

            if i % 200:
                logger.info("[%d, %5d] loss: %.3f" %
                            (epoch+1, i+1, running_loss/200))
                running_loss = 0.0
        logger.info('saving %dth ckp in %s.' % (epoch, ckp_path))
        torch.save(model.state_dict(), os.path.join(
            ckp_path, str(epoch)+".pth"))


if __name__ == "__main__":
    print("currently executing train.py file.")

    logger = Logger.Logger(time_stamp=False).createLogger()

    options = Options.OptionParser().parse()

    args = vars(options)
    logger.info("Options listed below:----------------")
    for k, v in args.items():
        logger.info("%s: %s" % (str(k), str(v)))
    logger.info("Options all listed.------------------")
