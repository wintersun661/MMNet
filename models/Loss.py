from numpy.core.fromnumeric import size
import torch.optim as optim
import torch.nn as nn


def createLoss(opt):
    if opt.loss_type == "BCE":
        # cross entropy loss
        criterion = nn.BCELoss(size_average=False)
    if opt.loss_type == "MSE":
        criterion = nn.MSELoss()
    return criterion
