#!/usr/bin/env python3
import torch
from torch import nn

from .inference import make_seg_postprocessor
from .loss import make_seg_loss_evaluator
import time


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


class SEGHead(nn.Module):
    """
    Adds a simple SEG Head with pixel-level prediction
    """

    def __init__(self, in_channels, cfg):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(SEGHead, self).__init__()
        self.cfg = cfg
        ndim = 256
        self.fpn_out5 = nn.Sequential(
            conv3x3(ndim, 64), nn.Upsample(scale_factor=8, mode="nearest")
        )
        self.fpn_out4 = nn.Sequential(
            conv3x3(ndim, 64), nn.Upsample(scale_factor=4, mode="nearest")
        )
        self.fpn_out3 = nn.Sequential(
            conv3x3(ndim, 64), nn.Upsample(scale_factor=2, mode="nearest")
        )
        self.fpn_out2 = conv3x3(ndim, 64)
        self.seg_out = nn.Sequential(
            conv3x3_bn_relu(in_channels, 64, 1),
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid(),
        )
        if self.cfg.MODEL.SEG.USE_PPM:
            # PPM Module
            pool_scales=(2, 4, 8)
            fc_dim = 256
            self.ppm_pooling = []
            self.ppm_conv = []
            for scale in pool_scales:
                self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
                self.ppm_conv.append(nn.Sequential(
                    nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True)
                ))
            self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
            self.ppm_conv = nn.ModuleList(self.ppm_conv)
            self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, ndim, 1)
            self.ppm_conv.apply(self.weights_init)
            self.ppm_last_conv.apply(self.weights_init)
        self.fpn_out5.apply(self.weights_init)
        self.fpn_out4.apply(self.weights_init)
        self.fpn_out3.apply(self.weights_init)
        self.fpn_out2.apply(self.weights_init)
        self.seg_out.apply(self.weights_init)

    def forward(self, x):
        if self.cfg.MODEL.SEG.USE_PPM:
            conv5 = x[-2]
            input_size = conv5.size()
            ppm_out = [conv5]
            for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
                ppm_out.append(pool_conv(nn.functional.interpolate(
                    pool_scale(conv5),
                    (input_size[2], input_size[3]),
                    mode='bilinear', align_corners=False)))
            ppm_out = torch.cat(ppm_out, 1)
            f = self.ppm_last_conv(ppm_out)
        else:
            f = x[-2]
        # p5 = self.fpn_out5(x[-2])
        p5 = self.fpn_out5(f)
        p4 = self.fpn_out4(x[-3])
        p3 = self.fpn_out3(x[-4])
        p2 = self.fpn_out2(x[-5])
        fuse = torch.cat((p5, p4, p3, p2), 1)
        out = self.seg_out(fuse)
        return out, fuse

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(1e-4)


class SEGModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(SEGModule, self).__init__()

        self.cfg = cfg.clone()

        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        head = SEGHead(in_channels, cfg)

        box_selector_train = make_seg_postprocessor(cfg, is_train=True)
        box_selector_test = make_seg_postprocessor(cfg, is_train=False)

        loss_evaluator = make_seg_loss_evaluator(cfg)

        # self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (Tensor): fused feature from FPN
            targets (Tensor): segmentaion gt map

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        preds, fuse_feature = self.head(features)
        # anchors = self.anchor_generator(images, features)
        image_shapes = images.get_sizes()
        if self.training:
            return self._forward_train(preds, targets, image_shapes), [fuse_feature]
        else:
            return self._forward_test(preds, image_shapes), [fuse_feature]

    def _forward_train(self, preds, targets, image_shapes):
        # Segmentation map must be transformed into boxes for detection.
        # sampled into a training batch.
        with torch.no_grad():
            boxes = self.box_selector_train(preds, image_shapes, targets)
        loss_seg = self.loss_evaluator(preds, targets)
        losses = {"loss_seg": loss_seg}
        return boxes, losses

    def _forward_test(self, preds, image_shapes):
        # torch.cuda.synchronize()
        # start_time = time.time()
        boxes, rotated_boxes, polygons, scores = self.box_selector_test(preds, image_shapes)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print('post time:', end_time - start_time)
        seg_results = {'rotated_boxes': rotated_boxes, 'polygons': polygons, 'preds': preds, 'scores': scores}
        return boxes, seg_results


def build_segmentation(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return SEGModule(cfg)
