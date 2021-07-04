import os
import torch
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2
from datetime import datetime
from matplotlib import gridspec

from utils import geometry


def _unnormalize(normalized_image):
    meanset = [0.485, 0.456, 0.406]
    stdset = [0.229, 0.224, 0.225]

    # in-place conversion
    for im_channel, mean, std in zip(normalized_image, meanset, stdset):
        im_channel.mul_(std).add_(mean)


def visualize_image_with_annotation(image, kps, bbox=None, visualization_path="./debug_visualized", suffix='tmp', normalized=False):

    # image shape: 3, height, width
    # kps shape: 2, valid_kps_num(10-20)
    if not os.path.isdir(visualization_path):
        os.makedirs(visualization_path)

    save_name = "annot"
    if suffix != "":
        save_name += "_"+suffix
    if normalized:
        _unnormalize(image)
    # clear canvas
    plt.cla()

    # output visualization materials
    if str(image.device) == "cuda:0":
        image = image.cpu().detach().numpy()

    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)

    for k in range(kps.shape[1]):
        x = float(kps[0, k])
        y = float(kps[1, k])

        c = np.random.rand(3)
        plt.gca().add_artist(plt.Circle((x, y), radius=5, color=c))

    plt.axis("off")

    # save figure at associated path
    plt.savefig(os.path.join(visualization_path, save_name),
                bbox_inches="tight", pad_inches=0.0)


def visualize_pred(data, pred, visualization_path='./debug_visualizated', suffix='tmp', idx='0', reverse=False):

    if not os.path.isdir(visualization_path):
        os.makedirs(visualization_path)

    save_name = "pred"
    if suffix != "":
        save_name += "_"+suffix
    save_name += "_"+idx

    src_prefix = 'src'
    trg_prefix = 'trg'
    if reverse:
        src_prefix = 'trg'
        trg_prefix = 'src'

    batchsize = len(data['src_img'])
    nrow = 5
    ncol = batchsize
    fig = plt.figure(figsize=(ncol+1, nrow+1))

    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=0.0, hspace=0.0,
                           top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1),
                           left=0.5/(ncol+1), right=1-0.5/(ncol+1)
                           )

    srcImg_List = []
    trgImg_List = []

    for i in range(batchsize):
        # avoid unexpected influences by in-place un-normalization
        srcImg = data[src_prefix+'_img'][i].clone()
        trgImg = data[trg_prefix+'_img'][i].clone()

        # back to original color
        _unnormalize(srcImg)
        _unnormalize(trgImg)

        srcImg = srcImg.permute(1, 2, 0)
        srcImg_List.append(srcImg)

        trgImg = trgImg.permute(1, 2, 0)
        trgImg_List.append(trgImg)

    # visualize original
    for i in range(batchsize):
        srcImg = srcImg_List[i]

        trgImg = trgImg_List[i]

        srcKps = data[src_prefix+'_kps'][i][:, :data['valid_kps_num'][i]]
        trgKps = data[trg_prefix+'_kps'][i][:, :data['valid_kps_num'][i]]

        jointImg = torch.cat((srcImg, trgImg), 0)

        plt.subplot(gs[0, i])

        for k in range(srcKps.shape[1]):
            xa = float(srcKps[0, k])
            ya = float(srcKps[1, k])
            xb = float(trgKps[0, k])
            yb = float(trgKps[1, k])+srcImg.shape[0]

            c = np.random.rand(3)
            plt.gca().add_artist(plt.Circle((xa, ya), radius=5, color=c))
            plt.gca().add_artist(plt.Circle((xb, yb), radius=5, color=c))

            plt.plot([xb, xa], [yb, ya], c=c, linestyle='-', linewidth=0.5)

        plt.imshow(jointImg.cpu().detach().numpy())
        plt.axis('off')

    # visualize correlation
    shift = 0
    if reverse:
        shift = 1
    for j in range(0, 4):
        for i in range(len(data['src_img'])):
            srcImg = srcImg_List[i]
            trgImg = trgImg_List[i]

            srcKps = data[src_prefix+'_kps'][i][:,
                                                :data['valid_kps_num'][i]]
            trgKps = data[trg_prefix+'_kps'][i][:,
                                                :data['valid_kps_num'][i]]
            print(srcImg.shape)
            prdKps = geometry.predict_kps(
                srcKps, pred[j][shift][i], originalShape=srcImg.shape[:2])
            jointImg = torch.cat((srcImg, trgImg), 0)
            jointImg = torch.cat((srcImg, trgImg), 0)

            plt.subplot(gs[j+1, i])
            plt.cla()
            for k in range(srcKps.shape[1]):
                xa = float(srcKps[0, k])
                ya = float(srcKps[1, k])
                xb = float(trgKps[0, k])
                yb = float(trgKps[1, k])+srcImg.shape[0]
                xc = float(prdKps[0][k])
                yc = float(prdKps[1][k])+srcImg.shape[0]

                c = np.random.rand(3)
                plt.gca().add_artist(plt.Circle((xa, ya), radius=5, color=c))
                plt.gca().add_artist(plt.Circle((xb, yb), radius=5, color=c))
                plt.gca().add_artist(plt.Circle((xc, yc), radius=5, color=c))
                plt.plot([xc, xa], [yc, ya], c=c, linestyle='-', linewidth=0.5)

            plt.imshow(jointImg.cpu().detach().numpy())
            plt.axis('off')

    plt.savefig(os.path.join(visualization_path, save_name),
                dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
