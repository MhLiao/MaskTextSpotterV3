#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..segmentation.segmentation import build_segmentation
from ..roi_heads.roi_heads import build_roi_heads
import time

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        if cfg.MODEL.SEG_ON:
            self.proposal = build_segmentation(cfg)
        else:
            self.proposal = build_rpn(cfg)
        if cfg.MODEL.TRAIN_DETECTION_ONLY:
            self.roi_heads = None
        else:
            self.roi_heads = build_roi_heads(cfg)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        # torch.cuda.synchronize()
        # start_time = time.time()
        images = to_image_list(images)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print('image load time:', end_time - start_time)
        # torch.cuda.synchronize()
        # start_time = time.time()
        features = self.backbone(images.tensors)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print('backbone time:', end_time - start_time)
        if self.cfg.MODEL.SEG_ON and not self.training:
            # torch.cuda.synchronize()
            # start_time = time.time()
            (proposals, seg_results), fuse_feature = self.proposal(images, features, targets)
            # torch.cuda.synchronize()
            # end_time = time.time()
            # print('seg time:', end_time - start_time)
        else:
            if self.cfg.MODEL.SEG_ON:
                (proposals, proposal_losses), fuse_feature = self.proposal(images, features, targets)
            else:
                proposals, proposal_losses = self.proposal(images, features, targets)
        if self.roi_heads is not None:
            if self.cfg.MODEL.SEG_ON and self.cfg.MODEL.SEG.USE_FUSE_FEATURE:
                x, result, detector_losses = self.roi_heads(fuse_feature, proposals, targets)
            else:
                x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            # x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            if self.roi_heads is not None:
                losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        else:
            if self.cfg.MODEL.SEG_ON:
                return result, proposals, seg_results
            else:
                return result

        # return result
