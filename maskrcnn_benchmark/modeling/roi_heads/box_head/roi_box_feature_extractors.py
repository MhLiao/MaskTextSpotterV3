#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.layers import Conv2d

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=has_bias
    )


def conv3x3_bn_relu(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
        )

        self.pooler = pooler
        self.head = head

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        super(FPN2MLPFeatureExtractor, self).__init__()
        self.cfg = cfg
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        if self.cfg.MODEL.ROI_BOX_HEAD.MIX_OPTION == 'CAT':
            input_size = (cfg.MODEL.BACKBONE.OUT_CHANNELS + 1) * resolution ** 2
        else:
            input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.pooler = pooler
        self.fc6 = nn.Linear(input_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        # if self.cfg.MODEL.ROI_BOX_HEAD.MIX_OPTION == 'ATTENTION':
        #     self.attention = nn.Sequential(
        #         conv3x3_bn_relu(cfg.MODEL.BACKBONE.OUT_CHANNELS + 1, 32),
        #         conv3x3(32, 1),
        #         nn.Sigmoid()
        #     )
        #     self.attention.apply(self.weights_init)
        # if self.cfg.MODEL.ROI_BOX_HEAD.MIX_OPTION == 'ATTENTION':
        #     self.attention = nn.Sequential(
        #         Conv2d(cfg.MODEL.BACKBONE.OUT_CHANNELS + 1, 1, 1, 1, 0),
        #         nn.Sigmoid()
        #     )
        #     for name, param in self.named_parameters():
        #         if "bias" in name:
        #             nn.init.constant_(param, 0)
        #         elif "weight" in name:
        #             # Caffe2 implementation uses MSRAFill, which in fact
        #             # corresponds to kaiming_normal_ in PyTorch
        #             nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

        for l in [self.fc6, self.fc7]:
            # Caffe2 implementation uses XavierFill, which in fact
            # corresponds to kaiming_uniform_ in PyTorch
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(1e-4)

    def feature_mask(self, x, proposals):
        masks = []
        for proposal in proposals:
            segmentation_masks = proposal.get_field("masks")
            boxes = proposal.bbox.to(torch.device("cpu"))
            for segmentation_mask, box in zip(segmentation_masks, boxes):
                cropped_mask = segmentation_mask.crop(box)
                scaled_mask = cropped_mask.resize((self.cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION, self.cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION))
                mask = scaled_mask.convert(mode="mask")
                masks.append(mask)
        if len(masks) == 0:
            if self.cfg.MODEL.ROI_BOX_HEAD.MIX_OPTION == 'CAT':
                x = cat([x, torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)], dim=1)
            return x
        masks = torch.stack(masks, dim=0).to(x.device, dtype=torch.float32)
        if self.cfg.MODEL.ROI_BOX_HEAD.MIX_OPTION == 'CAT':
            x = cat([x, masks.unsqueeze(1)], dim=1)
            return x
        if self.cfg.MODEL.ROI_BOX_HEAD.MIX_OPTION == 'ATTENTION':
            # x_cat = cat([x, masks.unsqueeze(1)], dim=1)
            # attention = self.attention(x_cat)
            # x = x * attention
            return x
        soft_ratio = self.cfg.MODEL.ROI_BOX_HEAD.SOFT_MASKED_FEATURE_RATIO
        if soft_ratio > 0:
            if soft_ratio < 1.0:
                x = x * (soft_ratio + (1 - soft_ratio) * masks.unsqueeze(1))
            else:
                x = x * (1.0 + soft_ratio * masks.unsqueeze(1))
        else:
            x = x * masks.unsqueeze(1)
        return x

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        if self.cfg.MODEL.ROI_BOX_HEAD.USE_MASKED_FEATURE:
            x = self.feature_mask(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


_ROI_BOX_FEATURE_EXTRACTORS = {
    "ResNet50Conv5ROIFeatureExtractor": ResNet50Conv5ROIFeatureExtractor,
    "FPN2MLPFeatureExtractor": FPN2MLPFeatureExtractor,
}


def make_roi_box_feature_extractor(cfg):
    func = _ROI_BOX_FEATURE_EXTRACTORS[cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR]
    return func(cfg)
