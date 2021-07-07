import numpy as np
import torch
import torch.nn as nn

from . import resnet
from functools import reduce
from operator import add
from torchvision import models

import torch.nn.functional as F

import gluoncvth as gcv


def crop(data1, data2, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    _, _, h2, w2 = data2.size()
    assert(h2 <= h1 and w2 <= w1)
    data = data1[:, :, crop_h:crop_h+h2, crop_w:crop_w+w2]
    return data


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class WeightAverage(nn.Module):
    def __init__(self, c_in, R=3):
        super(WeightAverage, self).__init__()
        c_out = c_in // 2

        self.conv_theta = nn.Conv2d(c_in, c_out, 1)
        self.conv_phi = nn.Conv2d(c_in, c_out, 1)
        self.conv_g = nn.Conv2d(c_in, c_out, 1)
        self.conv_back = nn.Conv2d(c_out, c_in, 1)
        self.CosSimLayer = nn.CosineSimilarity(dim=3)  # norm

        self.R = R
        self.c_out = c_out

    def forward(self, x):
        """
        x: torch.Tensor(batch_size, channel, h, w)
        """

        batch_size, c, h, w = x.size()
        padded_x = F.pad(x, (1, 1, 1, 1), 'replicate')
        neighbor = F.unfold(padded_x, kernel_size=self.R,
                            dilation=1, stride=1)  # BS, C*R*R, H*W
        neighbor = neighbor.contiguous().view(batch_size, c, self.R, self.R, h, w)
        neighbor = neighbor.permute(0, 2, 3, 1, 4, 5)  # BS, R, R, c, h ,w
        neighbor = neighbor.reshape(batch_size * self.R * self.R, c, h, w)

        theta = self.conv_theta(x)  # BS, C', h, w
        phi = self.conv_phi(neighbor)   # BS*R*R, C', h, w
        g = self.conv_g(neighbor)     # BS*R*R, C', h, w

        phi = phi.contiguous().view(batch_size, self.R, self.R, self.c_out, h, w)
        phi = phi.permute(0, 4, 5, 3, 1, 2)  # BS, h, w, c, R, R
        theta = theta.permute(0, 2, 3, 1).contiguous().view(
            batch_size, h, w, self.c_out)   # BS, h, w, c
        theta_dim = theta

        cos_sim = self.CosSimLayer(
            phi, theta_dim[:, :, :, :, None, None])  # BS, h, w, R, R

        softmax_sim = F.softmax(cos_sim.contiguous().view(
            batch_size, h, w, -1), dim=3).contiguous().view_as(cos_sim)  # BS, h, w, R, R

        g = g.contiguous().view(batch_size, self.R, self.R, self.c_out, h, w)
        g = g.permute(0, 4, 5, 1, 2, 3)  # BS, h, w, R, R, c_out

        weighted_g = g * softmax_sim[:, :, :, :, :, None]
        weighted_average = torch.sum(weighted_g.contiguous().view(
            batch_size, h, w, -1, self.c_out), dim=3)
        weight_average = weighted_average.permute(
            0, 3, 1, 2).contiguous()  # BS, c_out, h, w

        x_res = self.conv_back(weight_average)
        ret = x + x_res

        return ret


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(
            2).unsqueeze(3).expand_as(x) * x
        return out


class MSBlock(nn.Module):
    def __init__(self, c_in, rate=4):
        super(MSBlock, self).__init__()
        c_out = c_in
        self.rate = rate

        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        dilation = self.rate*1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1,
                               dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        dilation = self.rate*2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1,
                               dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        dilation = self.rate*3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1,
                               dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)

        self._initialize_weights()

    def forward(self, x):
        o = self.relu(self.conv(x))
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        out = o + o1 + o2 + o3
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class NonLocalMsBlock(nn.Module):
    def __init__(self, c_in, rate=4, R=3):
        super(NonLocalMsBlock, self).__init__()
        self.ms_block = MSBlock(c_in, rate)
        self.non_local_block = WeightAverage(c_in, R)

    def forward(self, x):
        return self.non_local_block(self.ms_block(x))


class MMNet(nn.Module):
    def __init__(self, logger=None, rate=4, device='cuda:0', backbone_net_name='resnet101'):
        super(MMNet, self).__init__()

        #self.L2Norm1 = L2Norm(21, 20)
        #self.L2Norm2 = L2Norm(21, 20)
        #self.L2Norm3 = L2Norm(21, 20)
        #self.L2Norm4 = L2Norm(21, 20)
        #self.pretrain = pretrain
        t = 1

        self.backbone_net_name = backbone_net_name

        self.device = device

        if self.backbone_net_name == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True).to(device)
            self.nbottlenecks = [3, 4, 6, 3]

        elif self.backbone_net_name == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True).to(device)
            self.nbottlenecks = [3, 4, 23, 3]

        elif self.backbone_net_name == 'resnext101':
            self.backbone = resnet.resnext101_32x8d(pretrained=True).to(device)
            self.nbottlenecks = [3, 4, 23, 3]

        elif self.backbone_net_name == 'fcn-resnet101':
            self.backbone = gcv.models.get_fcn_resnet101_voc(
                pretrained=True).to(device).pretrained
            self.nbottlenecks = [3, 4, 23, 3]

        #self.features = vgg16_c.VGG16_C(pretrain, logger)
        # changed to fit resnet101

        self.features = self.getResNetFeature_List

        self.msblock0 = MSBlock(64, rate)
        self.conv0_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv0_kernel = nn.Conv2d(21, 21, (3, 3), stride=1, padding=1)

        self.msblock1_1 = MSBlock(256, rate)
        self.msblock1_2 = MSBlock(256, rate)
        self.msblock1_3 = MSBlock(256, rate)
        self.conv1_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv1_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv1_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv1_kernel = nn.Conv2d(21, 21, (3, 3), stride=1, padding=1)
        self.conv1_scale = nn.Conv2d(42, 21, (3, 3), stride=1, padding=1)

        self.msblock2_1 = MSBlock(512, rate)
        self.msblock2_2 = MSBlock(512, rate)
        self.msblock2_3 = MSBlock(512, rate)
        self.msblock2_4 = MSBlock(512, rate)
        self.conv2_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv2_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv2_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv2_4_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv2_kernel = nn.Conv2d(21, 21, (3, 3), stride=1, padding=1)
        self.conv2_scale = nn.Conv2d(42, 21, (3, 3), stride=1, padding=1)

        self.msblock3_1 = MSBlock(1024, rate)
        self.msblock3_2 = MSBlock(1024, rate)
        self.msblock3_3 = MSBlock(1024, rate)
        self.msblock3_4 = MSBlock(1024, rate)
        self.msblock3_5 = MSBlock(1024, rate)
        self.msblock3_6 = MSBlock(1024, rate)
        self.msblock3_7 = MSBlock(1024, rate)
        self.msblock3_8 = MSBlock(1024, rate)
        self.msblock3_9 = MSBlock(1024, rate)
        self.msblock3_10 = MSBlock(1024, rate)
        self.msblock3_11 = MSBlock(1024, rate)
        self.msblock3_12 = MSBlock(1024, rate)
        self.msblock3_13 = MSBlock(1024, rate)
        self.msblock3_14 = MSBlock(1024, rate)
        self.msblock3_15 = MSBlock(1024, rate)
        self.msblock3_16 = MSBlock(1024, rate)
        self.msblock3_17 = MSBlock(1024, rate)
        self.msblock3_18 = MSBlock(1024, rate)
        self.msblock3_19 = MSBlock(1024, rate)
        self.msblock3_20 = MSBlock(1024, rate)
        self.msblock3_21 = MSBlock(1024, rate)
        self.msblock3_22 = MSBlock(1024, rate)
        self.msblock3_23 = MSBlock(1024, rate)
        self.conv3_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_4_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_5_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_6_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_7_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_8_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_9_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_10_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_11_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_12_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_13_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_14_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_15_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_16_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_17_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_18_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_19_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_20_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_21_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_22_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_23_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_kernel = nn.Conv2d(21, 21, (3, 3), stride=1, padding=1)
        self.conv3_scale = nn.Conv2d(42, 21, (3, 3), stride=1, padding=1)

        self.msblock4_1 = MSBlock(2048, rate)
        self.msblock4_2 = MSBlock(2048, rate)
        self.msblock4_3 = MSBlock(2048, rate)
        self.conv4_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv4_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv4_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv4_kernel = nn.Conv2d(21, 21, (3, 3), stride=1, padding=1)

        #self.upsample_2 = nn.ConvTranspose2d(4480, 17920, 4, stride=2, padding=1,bias=False)
        self.upsample_4 = nn.ConvTranspose2d(
            1120, 4480, 4, stride=2, padding=1, bias=False)
        self.upsample_8 = nn.ConvTranspose2d(
            280, 4480, 8, stride=4, padding=2, bias=False)
        self.upsample_16 = nn.ConvTranspose2d(
            70, 4480, 16, stride=8, padding=4, bias=False)

        self.feature_upsample_4 = nn.ConvTranspose2d(
            21, 21, 4, stride=2, padding=1, bias=False)
        self.feature_upsample_3 = nn.ConvTranspose2d(
            21, 21, 4, stride=2, padding=1, bias=False)
        self.feature_upsample_2 = nn.ConvTranspose2d(
            21, 21, 4, stride=2, padding=1, bias=False)

        #self.fuse = nn.Conv2d(17920, 4480, 1,  stride=1)
        self.fuse = nn.Linear(4, 1)

        # self._initialize_weights(logger)

        # self.non_local_0 = WeightAverage(21)
        self.non_local_1 = WeightAverage(21)
        self.non_local_2 = WeightAverage(21)
        self.non_local_3 = WeightAverage(21)
        self.non_local_4 = WeightAverage(21)

    def calLayer0(self, feats):
        sum0 = self.conv0_down(self.msblock0(feats[0]))
        # sum0 = self.non_local_0(sum0)
        return sum0

    def calLayer1(self, feats):
        sum1 = self.conv1_1_down(self.msblock1_1(feats[1])) + \
            self.conv1_2_down(self.msblock1_2(feats[2])) + \
            self.conv1_3_down(self.msblock1_3(feats[3]))
        sum1 = self.non_local_1(sum1)
        return sum1

    def calLayer2(self, feats):
        sum2 = self.conv2_1_down(self.msblock2_1(feats[4])) + \
            self.conv2_2_down(self.msblock2_2(feats[5])) +\
            self.conv2_3_down(self.msblock2_3(feats[6])) +\
            self.conv2_4_down(self.msblock2_4(feats[7]))
        sum2 = self.non_local_2(sum2)
        return sum2

    def _res50_calLayer3(self, feats):
        sum3 = self.conv3_1_down(self.msblock3_1(feats[8])) + \
            self.conv3_2_down(self.msblock3_2(feats[9])) + \
            self.conv3_3_down(self.msblock3_3(feats[10])) + \
            self.conv3_4_down(self.msblock3_4(feats[11])) + \
            self.conv3_5_down(self.msblock3_5(feats[12])) + \
            self.conv3_6_down(self.msblock3_6(feats[13]))
        sum3 = self.non_local_3(sum3)
        return sum3

    def calLayer3(self, feats):
        sum3 = self.conv3_1_down(self.msblock3_1(feats[8])) + \
            self.conv3_2_down(self.msblock3_2(feats[9])) + \
            self.conv3_3_down(self.msblock3_3(feats[10])) + \
            self.conv3_4_down(self.msblock3_4(feats[11])) + \
            self.conv3_5_down(self.msblock3_5(feats[12])) + \
            self.conv3_6_down(self.msblock3_6(feats[13])) + \
            self.conv3_7_down(self.msblock3_7(feats[14])) + \
            self.conv3_8_down(self.msblock3_8(feats[15])) + \
            self.conv3_9_down(self.msblock3_9(feats[16])) + \
            self.conv3_10_down(self.msblock3_10(feats[17])) + \
            self.conv3_11_down(self.msblock3_11(feats[18])) + \
            self.conv3_12_down(self.msblock3_12(feats[19])) + \
            self.conv3_13_down(self.msblock3_13(feats[20])) + \
            self.conv3_14_down(self.msblock3_14(feats[21])) + \
            self.conv3_15_down(self.msblock3_15(feats[22])) + \
            self.conv3_16_down(self.msblock3_16(feats[23])) + \
            self.conv3_17_down(self.msblock3_17(feats[24])) + \
            self.conv3_18_down(self.msblock3_18(feats[25])) + \
            self.conv3_19_down(self.msblock3_19(feats[26])) + \
            self.conv3_20_down(self.msblock3_20(feats[27])) + \
            self.conv3_21_down(self.msblock3_21(feats[28])) + \
            self.conv3_22_down(self.msblock3_22(feats[29])) + \
            self.conv3_23_down(self.msblock3_23(feats[30]))
        sum3 = self.non_local_3(sum3)
        return sum3

    def _res50_calLayer4(self, feats):
        sum4 = self.conv4_1_down(self.msblock4_1(feats[14])) + \
            self.conv4_2_down(self.msblock4_2(feats[15])) + \
            self.conv4_3_down(self.msblock4_3(feats[16]))
        sum4 = self.non_local_4(sum4)
        return sum4

    def calLayer4(self, feats):
        sum4 = self.conv4_1_down(self.msblock4_1(feats[31])) + \
            self.conv4_2_down(self.msblock4_2(feats[32])) + \
            self.conv4_3_down(self.msblock4_3(feats[33]))
        sum4 = self.non_local_4(sum4)
        return sum4

    def getResNetFeature_List(self, img, device='cuda:0'):

        feats = []

        nbottlenecks = self.nbottlenecks
        bottleneck_ids = reduce(
            add, list(map(lambda x: list(range(x)), nbottlenecks)))
        #print(self.bottleneck_ids) [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 0, 1, 2]
        layer_ids = reduce(
            add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        #print(self.layer_ids) [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4]

        # Layer 0
        feat = self.backbone.conv1.forward(img)

        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)

        feats.append(feat.clone())
        feat = self.backbone.maxpool.forward(feat)

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(bottleneck_ids, layer_ids)):

            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[
                bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[
                bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[
                bid].relu.forward(feat)

            feat = self.backbone.__getattr__('layer%d' % lid)[
                bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[
                bid].bn2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[
                bid].relu.forward(feat)

            feat = self.backbone.__getattr__('layer%d' % lid)[
                bid].conv3.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[
                bid].bn3.forward(feat)

            if bid == 0:
                res = self.backbone.__getattr__('layer%d' % lid)[
                    bid].downsample.forward(res)

            feat += res

            feat = self.backbone.__getattr__('layer%d' % lid)[
                bid].relu.forward(feat)
            feats.append(feat.clone())

        '''    
        # Up-sample & concatenate features to construct a hyperimage
        for idx, feat in enumerate(feats):
            if idx != 0:
                feats[idx] = F.interpolate(feat, tuple(feats[0].shape[2:]), None, 'bilinear', True)
        '''

        return feats

    def upsample(self, corr4d, ratio=2):
        n, h0, w0, h1, w1 = corr4d.shape
        corr4_reshaped = corr4d.view(n, h0*w0, h1, w1)
        corr4_upsampled = F.interpolate(
            corr4_reshaped, size=[h1*ratio, w1*ratio])
        corr4_reshaped = corr4_upsampled.permute(
            0, 2, 3, 1).contiguous().view(n, ratio*ratio*h1*w1, h0, w0)
        corr4_upsampled = F.interpolate(
            corr4_reshaped, size=[h0*ratio, w0*ratio])
        corr4AB_reshaped = corr4_upsampled.permute(0, 2, 3, 1).contiguous().view(
            n, ratio*h0, ratio*w0, ratio*h1, ratio*w1)
        return corr4AB_reshaped

    def forward(self, x):
        batch = len(x['src_img'])
        images = torch.cat((x['src_img'], x['trg_img']), 0)

        feats = self.features(images, device=self.device)

        features_src = []
        features_trg = []
        for i in feats:
            features_src.append(i[:batch])
            features_trg.append(i[batch:])

        if self.backbone_net_name in ["resnet101", "resnext101", 'fcn-resnet101']:
            calLayer4 = self.calLayer4
            calLayer3 = self.calLayer3
        elif self.backbone_net_name == "resnet50":
            calLayer4 = self._res50_calLayer4
            calLayer3 = self._res50_calLayer3
        calLayer2 = self.calLayer2
        calLayer1 = self.calLayer1

        # layer 4: 1/32 * 1/32 resolution 7*10

        sum4_src = calLayer4(features_src)
        sum4_trg = calLayer4(features_trg)
        # with shape: batchsize, 21, 7, 10

        # upsample features into higher resolution
        sum4_src_upsamples = self.feature_upsample_4(sum4_src)
        sum4_trg_upsamples = self.feature_upsample_4(sum4_trg)
        # with shape: batchsize, 21, 14, 20

        res_shape = sum4_src.shape[2:]
        # A->B source to target
        #corrMap4d_4_AB = torch.einsum('ijkl,ijmn->iklmn',[sum4_src, self.conv4_kernel(sum4_trg)])
        # B->A target to source
        #corrMap4d_4_BA = torch.einsum('ijkl,ijmn->iklmn',[sum4_trg, self.conv4_kernel(sum4_src)])
        # with shape: batchsize, 7, 10, 7, 10
        # print(corrMap4d_4_AB.shape,corrMap4d_4_BA.shape)
        corrMap4d_4_AB = torch.einsum('ijkl,ijmn->iklmn', [sum4_src, sum4_trg])
        corrMap4d_4_BA = torch.einsum('ijkl,ijmn->iklmn', [sum4_trg, sum4_src])
        # reshape
        pred4_AB_upsampled = self.upsample(corrMap4d_4_AB)
        pred4_BA_upsampled = self.upsample(corrMap4d_4_BA)

        #with shape: batchsize, 4480, 4480

        # layer 3: 1/16 * 1/16 resolution
        sum3_src = calLayer3(features_src)
        sum3_trg = calLayer3(features_trg)
        # with shape : batchsize, 21, 14, 20
        # concat deeper features into current layer
        sum3_src = torch.cat((sum3_src, sum4_src_upsamples), 1)
        sum3_trg = torch.cat((sum3_trg, sum4_trg_upsamples), 1)
        # with shape : batchsize, 42, 14, 20

        # conv
        sum3_src = self.conv3_scale(sum3_src)
        sum3_trg = self.conv3_scale(sum3_trg)
        #with shape: batchsize, 21, 14, 20

        # upsample to higher resolution
        sum3_src_upsamples = self.feature_upsample_3(sum3_src)
        sum3_trg_upsamples = self.feature_upsample_3(sum3_trg)

        res_shape = sum3_src.shape[2:]
        # A->B source to target
        corrMap4d_3_AB = torch.einsum('ijkl,ijmn->iklmn', [sum3_src, sum3_trg])
        # B->A target to source
        corrMap4d_3_BA = torch.einsum('ijkl,ijmn->iklmn', [sum3_trg, sum3_src])
        # with shape: batchsize, 14, 20, 14, 20

        # flat
        corrMap4d_3_AB = corrMap4d_3_AB + pred4_AB_upsampled.detach()
        corrMap4d_3_BA = corrMap4d_3_BA + pred4_BA_upsampled.detach()
        # print(corrMap4d_3_AB.shape,corrMap4d_3_BA.shape)
        pred3_AB_upsampled = self.upsample(corrMap4d_3_AB)
        pred3_BA_upsampled = self.upsample(corrMap4d_3_BA)

        # layer 2: 1/8 * 1/8 resolution
        sum2_src = calLayer2(features_src)
        sum2_trg = calLayer2(features_trg)
        # with shape : batchsize, 21, 28, 40
        # concat deeper features into current layer
        sum2_src = torch.cat((sum2_src, sum3_src_upsamples), 1)
        sum2_trg = torch.cat((sum2_trg, sum3_trg_upsamples), 1)
        # with shape : batchsize, 42, 28, 40

        # conv
        sum2_src = self.conv2_scale(sum2_src)
        sum2_trg = self.conv2_scale(sum2_trg)
        #with shape: batchsize, 21, 28, 40

        # upsample to higher resolution
        sum2_src_upsamples = self.feature_upsample_2(sum2_src)
        sum2_trg_upsamples = self.feature_upsample_2(sum2_trg)

        res_shape = sum2_src.shape[2:]
        # A->B source to target
        corrMap4d_2_AB = torch.einsum('ijkl,ijmn->iklmn', [sum2_src, sum2_trg])
        # B->A target to source
        corrMap4d_2_BA = torch.einsum('ijkl,ijmn->iklmn', [sum2_trg, sum2_src])
        # with shape: batchsize, 28, 40, 28, 40

        # flat
        corrMap4d_2_AB = corrMap4d_2_AB + pred3_AB_upsampled.detach()
        corrMap4d_2_BA = corrMap4d_2_BA + pred3_BA_upsampled.detach()
        # print(corrMap4d_2_AB.shape,corrMap4d_2_BA.shape)
        pred2_AB_upsampled = self.upsample(corrMap4d_2_AB)
        pred2_BA_upsampled = self.upsample(corrMap4d_2_BA)

        # layer 1 : 1/4 * 1/4 resolution
        sum1_src = calLayer1(features_src)
        sum1_trg = calLayer1(features_trg)
        # with shape: batchsize, 21,56,80
        # concat deeper features into current layer
        sum1_src = torch.cat((sum1_src, sum2_src_upsamples), 1)
        sum1_trg = torch.cat((sum1_trg, sum2_trg_upsamples), 1)
        # with shape : batchsize, 42, 56, 80

        # conv
        sum1_src = self.conv1_scale(sum1_src)
        sum1_trg = self.conv1_scale(sum1_trg)
        #with shape: batchsize, 21, 56, 80

        res_shape = sum1_src.shape[2:]
        # A->B source to target
        corrMap4d_1_AB = torch.einsum('ijkl,ijmn->iklmn', [sum1_src, sum1_trg])
        # B->A target to source
        corrMap4d_1_BA = torch.einsum('ijkl,ijmn->iklmn', [sum1_trg, sum1_src])
        # with shape: batchsize, 56, 80, 56, 80

        corrMap4d_1_AB = corrMap4d_1_AB + pred2_AB_upsampled.detach()
        corrMap4d_1_BA = corrMap4d_1_BA + pred2_BA_upsampled.detach()

        # fuse
        # difference between layers

        return [[corrMap4d_4_AB, corrMap4d_4_BA], [corrMap4d_3_AB, corrMap4d_3_BA], [corrMap4d_2_AB, corrMap4d_2_BA], [corrMap4d_1_AB, corrMap4d_1_BA]]

    def _initialize_weights(self, logger=None):
        for name, param in self.state_dict().items():
            if self.pretrain and 'features' in name:
                continue
            # elif 'down' in name:
            #     param.zero_()
            elif 'upsample' in name:
                if logger:
                    logger.info('init upsamle layer %s ' % name)
                k = int(name.split('.')[0].split('_')[1])
                param.copy_(get_upsampling_weight(1, 1, k*2))
            elif 'fuse' in name:
                if logger:
                    logger.info('init params %s ' % name)
                if 'bias' in name:
                    param.zero_()
                else:
                    nn.init.constant(param, 0.080)
            else:
                if logger:
                    logger.info('init params %s ' % name)
                if 'bias' in name:
                    param.zero_()
                else:
                    param.normal_(0, 0.01)
