r"""SPair-71k dataset"""
import json
import glob
import os

import numpy as np
import torch
from PIL import Image

from data import BaseDataset


def retrieveValidKps(anntn_pair):
    anntn_file_src, anntn_files_trg = anntn_pair
    t1 = anntn_file_src['kps']
    t2 = anntn_files_trg['kps']
    validIdxs = []

    for i in t1:
        if t1[i] is not None and t2[i] is not None:
            validIdxs.append(i)

    p1 = []
    p2 = []

    for i in validIdxs:
        p1.append(torch.Tensor([float(t1[i][0]), float(t1[i][1])]))
        p2.append(torch.Tensor([float(t2[i][0]), float(t2[i][1])]))
    return [torch.stack(p1).t(), torch.stack(p2).t()]


class CorrespondenceDataset(BaseDataset.CorrespondenceDataset):
    r"""Inherits CorrespondenceDataset"""

    def __init__(self, benchmark_name, data_path, thresh_type, split, device, resize_to, max_kps_num):
        r"""SPair-71k dataset constructor"""
        super(CorrespondenceDataset, self).__init__(
            benchmark_name, data_path, thresh_type, split, device, resize_to=resize_to, max_kps_num=max_kps_num)

        self.train_data = open(self.spt_path).read().split('\n')
        self.train_data = self.train_data[:len(self.train_data) - 1]
        self.src_imnames = list(
            map(lambda x: x.split('-')[1] + '.jpg', self.train_data))
        self.trg_imnames = list(map(lambda x: x.split(
            '-')[2].split(':')[0] + '.jpg', self.train_data))
        self.cls_names = list(map(lambda x: x.split(':')[1], self.train_data))
        self.cls = os.listdir(self.image_path)
        self.cls.sort()

        anntn_files_src = []
        anntn_files_trg = []
        for index in range(len(self.train_data)):
            #print(self.ann_path, self.cls_names[index], self.src_imnames[index][:-4])
            #print(glob.glob('%s/%s/%s.json' % (self.ann_path, self.cls_names[index], self.src_imnames[index][:-4])))
            #exit()
            anntn_files_src.append(glob.glob(
                '%s/%s/%s.json' % (self.ann_path, self.cls_names[index], self.src_imnames[index][:-4]))[0])
            anntn_files_trg.append(glob.glob(
                '%s/%s/%s.json' % (self.ann_path, self.cls_names[index], self.trg_imnames[index][:-4]))[0])

        anntn_files_src = list(
            map(lambda x: json.load(open(x)), anntn_files_src))
        anntn_files_trg = list(
            map(lambda x: json.load(open(x)), anntn_files_trg))

        self.anntn_files_pair = list(
            map(lambda x, y: [x, y], anntn_files_src, anntn_files_trg))

        self.cls_ids = list(
            map(lambda x: self.cls.index(x['category']), anntn_files_src))

    def __getitem__(self, idx):
        r"""Construct and return a batch for SPair-71k dataset"""
        sample = super(CorrespondenceDataset, self).__getitem__(idx)

        sample['pckthres'] = self.get_pckthres(sample).to(self.device)

        return sample

    def get_pckthres(self, sample):
        r"""Compute PCK threshold"""
        return super(CorrespondenceDataset, self).get_pckthres(sample)

    def get_points(self, index):
        r"""Return key-points of an image"""
        src_kps, trg_kps = retrieveValidKps(
            self.anntn_files_pair[index])
        src_box = torch.Tensor(self.anntn_files_pair[index][0]['bndbox'])
        trg_box = torch.Tensor(self.anntn_files_pair[index][1]['bndbox'])

        return src_kps, trg_kps, src_box, trg_box

    def get_image(self, img_names, idx):
        r"""return image tensor"""
        img_name = os.path.join(
            self.image_path, self.cls[self.cls_ids[idx]], img_names[idx])
        # get numpy version of image
        image = np.array(Image.open(img_name).convert("RGB"))
        # convert to tensor
        image = torch.tensor(image.transpose(2, 0, 1).astype(np.float32))
        return image
