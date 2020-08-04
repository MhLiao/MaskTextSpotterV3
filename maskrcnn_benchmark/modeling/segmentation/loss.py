#!/usr/bin/env python3
"""
This file contains specific functions for computing losses on the SEG
file
"""

import torch


class SEGLossComputation(object):
    """
    This class computes the SEG loss.
    """

    def __init__(self, cfg):
        self.eps = 1e-6
        self.cfg = cfg

    def __call__(self, preds, targets):
        """
        Arguments:
            preds (Tensor)
            targets (list[Tensor])
            masks (list[Tensor])
        Returns:
            seg_loss (Tensor)
        """
        image_size = (preds.shape[2], preds.shape[3])
        segm_targets, masks = self.prepare_targets(targets, image_size)
        device = preds.device
        segm_targets = segm_targets.float().to(device)
        masks = masks.float().to(device)
        seg_loss = self.dice_loss(preds, segm_targets, masks)
        return seg_loss

    def dice_loss(self, pred, gt, m):
        intersection = torch.sum(pred * gt * m)
        union = torch.sum(pred * m) + torch.sum(gt * m) + self.eps
        loss = 1 - 2.0 * intersection / union
        return loss

    def project_masks_on_image(self, mask_polygons, labels, shrink_ratio, image_size):
        seg_map, training_mask = mask_polygons.convert_seg_map(
            labels, shrink_ratio, image_size, self.cfg.MODEL.SEG.IGNORE_DIFFICULT
        )
        return torch.from_numpy(seg_map), torch.from_numpy(training_mask)

    def prepare_targets(self, targets, image_size):
        segms = []
        training_masks = []
        for target_per_image in targets:
            segmentation_masks = target_per_image.get_field("masks")
            labels = target_per_image.get_field("labels")
            seg_maps_per_image, training_masks_per_image = self.project_masks_on_image(
                segmentation_masks, labels, self.cfg.MODEL.SEG.SHRINK_RATIO, image_size
            )
            segms.append(seg_maps_per_image)
            training_masks.append(training_masks_per_image)
        return torch.stack(segms), torch.stack(training_masks)


def make_seg_loss_evaluator(cfg):
    loss_evaluator = SEGLossComputation(cfg)
    return loss_evaluator
