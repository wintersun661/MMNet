"""Provides memory buffer and logger for evaluation"""

import logging

from skimage import draw
import numpy as np
import torch


def eval_pck(prd_kps, data, idx):
    r"""Compute percentage of correct key-points (PCK) based on prediction"""
    pckthres = data['pckthres'][idx]
    ncorrt = correct_kps(data['trg_kps'][idx][:, :data['valid_kps_num'][idx]].cuda(
    ), prd_kps, pckthres, data['alpha'])
    pair_pck = int(ncorrt) / int(data['valid_kps_num'][idx])

    return pair_pck


def correct_kps(trg_kps, prd_kps, pckthres, alpha=0.1):
    r"""Compute the number of correctly transferred key-points"""

    l2dist = torch.pow(torch.sum(torch.pow(trg_kps - prd_kps, 2), 0), 0.5)
    thres = pckthres.expand_as(l2dist).float()
    correct_pts = torch.le(l2dist, thres * alpha)
    
   
    return torch.sum(correct_pts)
