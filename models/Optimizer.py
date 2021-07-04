import torch
import torch.optim as optim
import re


def createOptimizer(opt, net):
    params = []

    base_lr = opt.lr
    weight_decay = opt.weight_decay

    params_dict = dict(net.named_parameters())
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

    if opt.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(params, momentum=opt.momentum,
                                    lr=opt.lr, weight_decay=opt.weight_decay)

    return optimizer
