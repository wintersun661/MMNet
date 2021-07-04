import os
import torch
import numpy as np
from torchvision import transforms

from PIL import Image
#import imgaug.augmenters as iaa

import torch.nn.functional as F

import csv


def readFile(filePath):
    with open(filePath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        feature_dict = []
        data_list = []
        for row in csv_reader:
            if line_count == 0:
                for name in row:
                    feature_dict.append(name)
                line_count += 1
            else:
                data_list.append(row)

    return feature_dict, data_list


class Normalize():
    r"""Image Normalization"""

    def __init__(self, image_keys, norm_range=True):
        self.image_keys = image_keys
        self.norm_range = norm_range
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        for key in self.image_keys:
            if self.norm_range:
                sample[key] /= 255.0
            # in-place normalization
            sample[key] = self.normalize(sample[key])
        return sample


class UnNormalize():
    r"""Image Un-normalization"""

    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image):
        img = image.clone()
        for im_channel, mean, std in zip(img, self.mean, self.std):
            im_channel.mul_(std).add_(mean)
        return img


class CorrespondenceDataset(torch.utils.data.Dataset):

    def __init__(self, benchmark_name, data_path, thresh_type, split, device, resize_to="", max_kps_num=5):
        super(CorrespondenceDataset, self).__init__()

        self.meta_data = {
            "pfpascal": ("PF-PASCAL", "_pairs.csv", "JPEGImages", "Annotations", "img"),
            "spair": ("SPair-71k", "Layout/large", "JPEGImages", "ImageAnnotation", "bbox"),
        }

        self.benchmark_name = benchmark_name
        self.device = device
        self.split_name = split
        self.max_kps_num = max_kps_num

        # directory path
        base_path = os.path.join(os.path.abspath(
            data_path), self.meta_data[self.benchmark_name][0])

        self.ann_path = os.path.join(
            base_path, self.meta_data[self.benchmark_name][3])
        self.image_path = os.path.join(
            base_path, self.meta_data[self.benchmark_name][2])

        # customized processing
        if benchmark_name == "pfpascal":
            self.spt_path = os.path.join(base_path, str(
                split)+self.meta_data[self.benchmark_name][1])
        if benchmark_name == "spair":
            self.spt_path = os.path.join(
                base_path, self.meta_data[self.benchmark_name][1], split+'.txt')

        # miscellaneous
        self.thresh_type = self.meta_data[self.benchmark_name][4] if thresh_type == "auto" else thresh_type
        self.transform = Normalize(["src_img", "trg_img"])

        self.resize_flag = False
        # whether to resize input image
        if resize_to != "":
            self.resize_flag = True
            height, width = resize_to.split(',')
            self.target_width = int(width)
            self.target_height = int(height)

        # initialized list in subclass constructors
        self.src_imnames = []
        self.trg_imnames = []
        self.src_kps = []
        self.trg_kps = []
        self.cls_ids = []
        self.cls = []
        self.train_data = []

    def __len__(self):
        r"""return the number of pairs"""
        return len(self.train_data)

    def __getitem__(self, index):
        r"""construct and return a batch"""

        sample = {}

        # image names
        sample["src_imname"] = self.src_imnames[index]
        sample["trg_imname"] = self.trg_imnames[index]

        # class of instances
        sample["pair_classid"] = self.cls_ids[index]
        sample["pair_class"] = self.cls[sample["pair_classid"]]

        # image tensors
        sample["src_img"] = self.get_image(
            self.src_imnames, index)
        sample["trg_img"] = self.get_image(
            self.trg_imnames, index)

        # keypoints tensors
        sample["src_kps"] = self.get_points(
            self.src_kps, index)
        sample["trg_kps"] = self.get_points(
            self.trg_kps, index)
        sample["valid_kps_num"] = len(sample["src_kps"][0])

        # perform image normalization with imagenet mean & std
        if self.transform:
            sample = self.transform(sample)

        # perform resize operation when specified
        if self.resize_flag:
            sample["src_img"], sample["src_kps"] = self.resize(
                sample["src_img"], sample["src_kps"])
            sample["trg_img"], sample["trg_kps"] = self.resize(
                sample["trg_img"], sample["trg_kps"])

        sample["src_img"] = sample["src_img"].to(self.device)
        sample["trg_img"] = sample["trg_img"].to(self.device)
        sample["src_kps"] = sample["src_kps"].to(self.device)
        sample["trg_kps"] = sample["trg_kps"].to(self.device)

        sample["datalen"] = len(self.src_imnames)

        return sample

    def get_image(self, img_names, idx):
        r"""return image tensor"""
        img_name = os.path.join(self.image_path, img_names[idx])
        # get numpy version of image
        image = np.array(Image.open(img_name).convert("RGB"))
        # convert to tensor
        image = torch.tensor(image.transpose(2, 0, 1).astype(np.float32))
        return image

    def get_points(self, pts, index):
        return pts[index]

    def resize(self, image, kps):
        r"""jointly resize image and correspond keypoints values"""
        image = image.unsqueeze(0)
        orig_size = image.shape

        inter_ratio_h = float(self.target_height)/orig_size[2]
        inter_ratio_w = float(self.target_width)/orig_size[3]

        image = F.interpolate(image, size=(
            self.target_height, self.target_width), mode="bilinear", align_corners=False).squeeze(0)
        kps[0, :] *= inter_ratio_w
        kps[1, :] *= inter_ratio_h

        padded_kps = torch.zeros(2, self.max_kps_num)
        n = kps.shape[1]
        padded_kps[:, :n] = kps

        return image, padded_kps

    def get_pckthres(self, sample):
        r"""Compute PCK threshold"""
        if self.thresh_type == 'bbox':
            trg_bbox = sample['trg_bbox']
            return torch.max(trg_bbox[2]-trg_bbox[0], trg_bbox[3]-trg_bbox[1])
        elif self.thresh_type == 'img':
            return torch.tensor(max(sample['trg_img'].size(1), sample['trg_img'].size(2)))
        else:
            raise Exception('Invalid pck evaluation level: %s' % self.thres)
