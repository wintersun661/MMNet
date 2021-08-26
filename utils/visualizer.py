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
from matplotlib.patches import Rectangle


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

    ax = plt.gca()
    # output visualization materials
    if str(image.device) == "cuda:0":
        image = image.cpu().detach().numpy()

    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)

    for k in range(kps.shape[1]):
        x = float(kps[0, k])
        y = float(kps[1, k])

        c = np.random.rand(3)
        ax.add_artist(plt.Circle((x, y), radius=5, color=c))

    if bbox != None:
        from matplotlib.patches import Rectangle
        rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3] -
                         bbox[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis("off")

    # save figure at associated path
    plt.savefig(os.path.join(visualization_path, save_name),
                bbox_inches="tight", pad_inches=0.0)


def visualize_pred(data, pred, visualization_path='./debug_visualizated', suffix='tmp', idx='0', reverse=False):

    if not os.path.isdir(visualization_path):
        os.makedirs(visualization_path)

    w, h = data['src_img'][0].shape[1:3]

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

    ncol = 5
    nrow = batchsize

    fig = plt.figure(figsize=(ncol, nrow+1))

    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=0.0, hspace=0.0,
                           top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1),
                           #    left=0.5/(ncol+1), right=1-0.5/(ncol+1)
                           )

    srcImg_List = []
    trgImg_List = []

    def __plot_star(xa, ya, line_width, offset, c):
        plt.plot([xa-offset, ya], [xa+offset, ya], c=c,
                 linestyle='-', linewidth=line_width)
        plt.plot([xa, ya-offset], [xa, ya+offset], c=c,
                 linestyle='-', linewidth=line_width)

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

        plt.subplot(gs[i, 0])
        plt.cla()
        ax = plt.gca()
        for k in range(srcKps.shape[1]):
            xa = float(srcKps[0, k])
            ya = float(srcKps[1, k])
            xb = float(trgKps[0, k])
            yb = float(trgKps[1, k])+srcImg.shape[0]

            c = np.random.rand(3)
            ax.add_artist(plt.Circle((xa, ya), radius=0.5, color=c))
            ax.add_artist(plt.Circle((xb, yb), radius=0.5, color=c))

            # plt.plot([xb, xa], [yb, ya], c=c, linestyle='-', linewidth=0.5)
        bbox_src = data['src_bbox'][i]
        bbox_trg = data['trg_bbox'][i]
        # rect = Rectangle((bbox_src[0],bbox_src[1]),bbox_src[2]-bbox_src[0],bbox_src[3]-bbox_src[1],linewidth=1,edgecolor='r',facecolor='none')
        # ax.add_patch(rect)
        # rect = Rectangle((bbox_trg[0],bbox_trg[1]+srcImg.shape[0]),bbox_trg[2]-bbox_trg[0],bbox_trg[3]-bbox_trg[1],linewidth=1,edgecolor='r',facecolor='none')
        # ax.add_patch(rect)

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

            prdKps = geometry.predict_kps(
                srcKps, pred[j][shift][i], originalShape=[h, w])
            jointImg = torch.cat((srcImg, trgImg), 0)
            jointImg = torch.cat((srcImg, trgImg), 0)

            plt.subplot(gs[i, j+1])
            plt.cla()
            for k in range(srcKps.shape[1]):
                xa = float(srcKps[0, k])
                ya = float(srcKps[1, k])
                xb = float(trgKps[0, k])
                yb = float(trgKps[1, k])+srcImg.shape[0]
                xc = float(prdKps[0][k])
                yc = float(prdKps[1][k])+srcImg.shape[0]

                c = np.random.rand(3)
                plt.gca().add_artist(plt.Circle((xa, ya), radius=0.5, color=c))
                plt.gca().add_artist(plt.Circle((xb, yb), radius=0.5, color=c))
                plt.gca().add_artist(plt.Circle((xc, yc), radius=0.5, color=c))
                # plt.plot([xc, xa], [yc, ya], c=c, linestyle='-', linewidth=0.5)

            plt.imshow(jointImg.cpu().detach().numpy())
            plt.axis('off')

    plt.savefig(os.path.join(visualization_path, save_name),
                dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def vis_corr(src_img_tensor, trg_img_tensor, src_kps, trg_kps, prd_kps, save_path, idx):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.cla()

    meanset = [0.485, 0.456, 0.406]
    stdset = [0.229, 0.224, 0.225]

    radis_size = 2.5
    line_width = 0.5
    max_kps_number = 60
    colors = ['indianred', 'orange', 'green', 'navy', 'maroon', 'darkorange', 'darkgreen', 'darkblue', 'darkred', 'olive', 'seagreen', 'midnightblue', 'firebrick', 'gold', 'darkcyan', 'darkslateblue',
              'orangered', 'teal', 'purple', 'coral', 'lawngreen', 'skyblue', 'hotpink', 'tomato', 'mediumseagreen', 'aqua', 'magenta', 'lightcoral', 'palegreen', 'steelblue', 'deeppink', 'darksalmon', 'springgreen']

    nrows = 1
    ncols = 1
    fig = plt.figure(figsize=(nrows+1, ncols+1))

    src_img = src_img_tensor.clone()
    for im_channel, mean, std in zip(src_img, meanset, stdset):
        im_channel.mul_(std).add_(mean)
    src_img = src_img.permute(1, 2, 0)

    trg_img = trg_img_tensor.clone()
    for im_channel, mean, std in zip(trg_img, meanset, stdset):
        im_channel.mul_(std).add_(mean)
    trg_img = trg_img.permute(1, 2, 0)

    def __plot_star(xa, ya, line_width, offset, c):
        plt.plot([xa-offset, ya], [xa+offset, ya], c=c,
                 linestyle='-', linewidth=line_width)
        plt.plot([xa, ya-offset], [xa, ya+offset], c=c,
                 linestyle='-', linewidth=line_width)

    def __sub_plot_corr(src_img, trg_img, src_kps, trg_kps, colors, radis_size, line_width, nrows, ncols, index, truth_kps=None):
        # if(src_img.shape[:2] != trg_img.shape[:2]): return
        joint_img = torch.cat((src_img, trg_img), 0)
        kps_num = len(src_kps[0])

        plt.subplot(nrows, ncols, index)
        for k in range(kps_num):
            xa = float(src_kps[0, k])
            ya = float(src_kps[1, k])
            xb = float(trg_kps[0, k])
            yb = float(trg_kps[1, k]) + src_img.shape[0]

            if not(xa > 0 and ya > 0):
                continue

            c = colors[k]
            plt.gca().add_artist(plt.Circle((xa, ya), radius=radis_size, color=c))
            plt.gca().add_artist(plt.Circle((xb, yb), radius=radis_size, color=c))

            xt = float(truth_kps[0, k])
            yt = float(truth_kps[1, k]) + src_img.shape[0]
            # plt.gca().add_artist(plt.Circle((xt,yt), radius=radis_size, color=c))
            line_len = 5.5
            plt.plot([xt-line_len, xt+line_len], [yt, yt], c=c,
                     linestyle='-', linewidth=radis_size/3.8)
            plt.plot([xt, xt], [yt-line_len, yt+line_len], c=c,
                     linestyle='-', linewidth=radis_size/3.8)

            plt.plot([xb, xa], [yb, ya], c=c,
                     linestyle='-', linewidth=line_width)

        plt.imshow(joint_img.cpu().detach().numpy())
        plt.axis('off')

    __sub_plot_corr(src_img, trg_img, src_kps, prd_kps, colors,
                    radis_size, line_width, nrows, ncols, 1, trg_kps)

    plt.subplots_adjust(hspace=0.00, wspace=0.00)
    plt.savefig(save_path+'/val'+str(idx) + '.jpg',
                dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
