from data import BaseDataset

import os
import scipy.io as sio
import pandas as pd
import numpy as np

import torch
from utils import visualizer


def read_mat(path, obj_name):
    r"""Read specified objects from Matlab data file, (.mat)"""
    mat_contents = sio.loadmat(path)
    mat_obj = mat_contents[obj_name]

    return mat_obj


class CorrespondenceDataset(BaseDataset.CorrespondenceDataset):
    r"""inherits correspondence dataset base"""

    def __init__(self, benchmark_name, data_path, thresh_type, split, device, resize_to, max_kps_num):
        super(CorrespondenceDataset, self).__init__(benchmark_name, data_path,
                                                    thresh_type, split, device, resize_to=resize_to, max_kps_num=max_kps_num)

        self.cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        self.train_data = pd.read_csv(self.spt_path)

        self.src_imnames = np.array(self.train_data.iloc[:, 0])
        self.trg_imnames = np.array(self.train_data.iloc[:, 1])

        self.cls_ids = self.train_data.iloc[:, 2].values.astype("int") - 1

        if self.split_name == "trn":
            self.flip = self.train_data.iloc[:, 3].values.astype("int")

        self.src_bbox = []
        self.trg_bbox = []

        for src_imname, trg_imname, cls in zip(self.src_imnames, self.trg_imnames, self.cls_ids):
            src_anns = os.path.join(self.ann_path, self.cls[cls],
                                    os.path.basename(src_imname))[:-4] + '.mat'
            trg_anns = os.path.join(self.ann_path, self.cls[cls],
                                    os.path.basename(trg_imname))[:-4] + '.mat'

            src_kp = torch.tensor(read_mat(src_anns, 'kps')).float()
            trg_kp = torch.tensor(read_mat(trg_anns, 'kps')).float()
            src_box = torch.tensor(read_mat(src_anns, 'bbox')[0].astype(float))
            trg_box = torch.tensor(read_mat(trg_anns, 'bbox')[0].astype(float))

            src_kps = []
            trg_kps = []
            for src_kk, trg_kk in zip(src_kp, trg_kp):
                if len(torch.isnan(src_kk).nonzero()) != 0 or \
                        len(torch.isnan(trg_kk).nonzero()) != 0:
                    continue
                else:
                    src_kps.append(src_kk)
                    trg_kps.append(trg_kk)
            self.src_kps.append(torch.stack(src_kps).t())
            self.trg_kps.append(torch.stack(trg_kps).t())
            self.src_bbox.append(src_box)
            self.trg_bbox.append(trg_box)

        self.src_imnames = list(
            map(lambda x: os.path.basename(x), self.src_imnames))
        self.trg_imnames = list(
            map(lambda x: os.path.basename(x), self.trg_imnames))

    def __getitem__(self, index):
        sample = super(CorrespondenceDataset, self).__getitem__(index)

        sample['src_bbox'] = self.src_bbox[index].to(self.device)
        sample['trg_bbox'] = self.trg_bbox[index].to(self.device)
        sample['pckthres'] = self.get_pckthres(sample).to(self.device)

        # Horizontal flip of key-points when training
        if self.split_name == 'trn' and self.flip[index]:
            sample['src_kps'][0] = sample['src_img'].size()[2] - \
                sample['src_kps'][0]
            sample['trg_kps'][0] = sample['trg_img'].size()[2] - \
                sample['trg_kps'][0]

        # visualizer.visualize_image_with_annotation(
        #     sample['src_img'], sample['src_kps'][:, :sample["valid_kps_num"]], sample['src_bbox'], normalized=True, suffix="src")

        return sample

    def get_pckthres(self, sample):
        r"""Compute PCK threshold"""
        return super(CorrespondenceDataset, self).get_pckthres(sample)