#!/usr/bin/env python3
import numpy as np
import torch
import cv2
import pyclipper
from shapely.geometry import Polygon

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist, cat_boxlist_gt
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
import random

import time


class SEGPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(
        self,
        top_n,
        binary_thresh,
        box_thresh,
        min_size,
        cfg,
    ):
        """
        Arguments:
            top_n (int)
            binary_thresh (float)
            box_thresh (float)
            min_size (int)
        """
        super(SEGPostProcessor, self).__init__()
        self.top_n = top_n
        self.binary_thresh = binary_thresh
        self.box_thresh = box_thresh
        self.min_size = min_size
        self.cfg = cfg

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        # device = proposals[0].bbox.
        if self.cfg.MODEL.SEG.USE_SEG_POLY or self.cfg.MODEL.ROI_BOX_HEAD.USE_MASKED_FEATURE or self.cfg.MODEL.ROI_MASK_HEAD.USE_MASKED_FEATURE:
            gt_boxes = [target.copy_with_fields(['masks']) for target in targets]
        else:
            gt_boxes = [target.copy_with_fields([]) for target in targets]
        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        # for gt_box in gt_boxes:
        #     gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))
        proposals = [
            cat_boxlist_gt([proposal, gt_box])
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def aug_tensor_proposals(self, boxes):
        # boxes: N * 4
        boxes = boxes.float()
        N = boxes.shape[0]
        device = boxes.device
        aug_boxes = torch.zeros((4, N, 4), device=device)
        aug_boxes[0, :, :] = boxes.clone()
        xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x_center = (xmin + xmax) / 2.
        y_center = (ymin + ymax) / 2.
        width = xmax - xmin
        height = ymax - ymin
        for i in range(3):
            choice = random.random()
            if choice < 0.5:
                # shrink or expand
                ratio = (torch.randn((N,), device=device) * 3 + 1) / 2.
                height = height * ratio
                ratio = (torch.randn((N,), device=device) * 3 + 1) / 2.
                width = width * ratio
            else:
                move_x = width * (torch.randn((N,), device=device) * 4 - 2)
                move_y = height * (torch.randn((N,), device=device) * 4 - 2)
                x_center += move_x
                y_center += move_y
            boxes[:, 0] = x_center - width / 2
            boxes[:, 2] = x_center + width / 2
            boxes[:, 1] = y_center - height / 2
            boxes[:, 3] = y_center + height / 2
            aug_boxes[i+1, :, :] = boxes.clone()
        return aug_boxes.reshape((-1, 4))

    def forward_for_single_feature_map(self, pred, image_shapes):
        """
        Arguments:
            pred: tensor of size N, 1, H, W
        """
        device = pred.device
        # torch.cuda.synchronize()
        # start_time = time.time()
        bitmap = self.binarize(pred)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print('binarize time:', end_time - start_time)
        N, height, width = pred.shape[0], pred.shape[2], pred.shape[3]
        # torch.cuda.synchronize()
        # start_time = time.time()
        bitmap_numpy = bitmap.cpu().numpy()  # The first channel
        pred_map_numpy = pred.cpu().numpy()
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print('gpu2numpy time:', end_time - start_time)
        boxes_batch = []
        rotated_boxes_batch = []
        polygons_batch = []
        scores_batch = []
        # torch.cuda.synchronize()
        # start_time = time.time()
        for batch_index in range(N):
            image_shape = image_shapes[batch_index]
            boxes, scores, rotated_boxes, polygons = self.boxes_from_bitmap(
                pred_map_numpy[batch_index],
                bitmap_numpy[batch_index], width, height)
            boxes = boxes.to(device)
            if self.training and self.cfg.MODEL.SEG.AUG_PROPOSALS:
                boxes = self.aug_tensor_proposals(boxes)
            if boxes.shape[0] > self.top_n:
                boxes = boxes[:self.top_n, :]
                # _, top_index = scores.topk(self.top_n, 0, sorted=False)
                # boxes = boxes[top_index, :]
                # scores = scores[top_index]
            # boxlist = BoxList(boxes, (width, height), mode="xyxy")
            boxlist = BoxList(boxes, (image_shape[1], image_shape[0]), mode="xyxy")
            if self.cfg.MODEL.SEG.USE_SEG_POLY or self.cfg.MODEL.ROI_BOX_HEAD.USE_MASKED_FEATURE or self.cfg.MODEL.ROI_MASK_HEAD.USE_MASKED_FEATURE:
                masks = SegmentationMask(polygons, (image_shape[1], image_shape[0]))
                boxlist.add_field('masks', masks)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            # boxlist = remove_small_boxes(boxlist, self.min_size)
            boxes_batch.append(boxlist)
            rotated_boxes_batch.append(rotated_boxes)
            polygons_batch.append(polygons)
            scores_batch.append(scores)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print('loop time:', end_time - start_time)
        return boxes_batch, rotated_boxes_batch, polygons_batch, scores_batch

    def forward(self, seg_output, image_shapes, targets=None):
        """
        Arguments:
            seg_output: list[tensor]

        Returns:
            boxlists (list[BoxList]): bounding boxes
        """
        sampled_boxes = []
        boxes_batch, rotated_boxes_batch, polygons_batch, scores_batch = self.forward_for_single_feature_map(seg_output, image_shapes)
        if not self.training:
            return boxes_batch, rotated_boxes_batch, polygons_batch, scores_batch
        sampled_boxes.append(boxes_batch)

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)
        return boxlists

    # def select_over_all_levels(self, boxlists):
    #     num_images = len(boxlists)
    #     # different behavior during training and during testing:
    #     # during training, post_nms_top_n is over *all* the proposals combined, while
    #     # during testing, it is over the proposals for each image
    #     # TODO resolve this difference and make it consistent. It should be per image,
    #     # and not per batch
    #     if self.training:
    #         objectness = torch.cat(
    #             [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
    #         )
    #         box_sizes = [len(boxlist) for boxlist in boxlists]
    #         post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
    #         _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
    #         inds_mask = torch.zeros_like(objectness, dtype=torch.uint8)
    #         inds_mask[inds_sorted] = 1
    #         inds_mask = inds_mask.split(box_sizes)
    #         for i in range(num_images):
    #             boxlists[i] = boxlists[i][inds_mask[i]]
    #     else:
    #         for i in range(num_images):
    #             objectness = boxlists[i].get_field("objectness")
    #             post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
    #             _, inds_sorted = torch.topk(
    #                 objectness, post_nms_top_n, dim=0, sorted=True
    #             )
    #             boxlists[i] = boxlists[i][inds_sorted]
    #     return boxlists

    def binarize(self, pred):
        if self.cfg.MODEL.SEG.USE_MULTIPLE_THRESH:
            binary_maps = []
            for thre in self.cfg.MODEL.SEG.MULTIPLE_THRESH:
                binary_map = pred > thre
                binary_maps.append(binary_map)
            return torch.cat(binary_maps, dim=1)
        else:
            return pred > self.binary_thresh

    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        """
        # assert _bitmap.size(0) == 1
        # bitmap = _bitmap[0]  # The first channel
        pred = pred[0]
        height, width = bitmap.shape[1], bitmap.shape[2]
        boxes = []
        scores = []
        rotated_boxes = []
        polygons = []
        contours_all = []
        for i in range(bitmap.shape[0]):
            try:
                _, contours, _ = cv2.findContours(
                    (bitmap[i] * 255).astype(np.uint8),
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_NONE,
                )
            except BaseException:
                contours, _ = cv2.findContours(
                    (bitmap[i] * 255).astype(np.uint8),
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_NONE,
                )
            contours_all.extend(contours)
        for contour in contours_all:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygon = approx.reshape((-1, 2))
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points)
            if not self.training and self.box_thresh > score:
                continue
            if polygon.shape[0] > 2:
                polygon = self.unclip(polygon, expand_ratio=self.cfg.MODEL.SEG.EXPAND_RATIO)
                if len(polygon) > 1:
                    continue
            else:
                continue
            # polygon = polygon.reshape(-1, 2)
            polygon = polygon.reshape(-1)
            box = self.unclip(points, expand_ratio=self.cfg.MODEL.SEG.BOX_EXPAND_RATIO).reshape(-1, 2)
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height
            )
            min_x, min_y = min(box[:, 0]), min(box[:, 1])
            max_x, max_y = max(box[:, 0]), max(box[:, 1])
            horizontal_box = torch.from_numpy(np.array([min_x, min_y, max_x, max_y]))
            boxes.append(horizontal_box)
            scores.append(score)
            rotated_box, _ = self.get_mini_boxes(box.reshape(-1, 1, 2))
            rotated_box = np.array(rotated_box)
            rotated_boxes.append(rotated_box)
            polygons.append([polygon])
        if len(boxes) == 0:
            boxes = [torch.from_numpy(np.array([0, 0, 0, 0]))]
            scores = [0.]

        boxes = torch.stack(boxes)
        scores = torch.from_numpy(np.array(scores))
        return boxes, scores, rotated_boxes, polygons

    def aug_proposals(self, box):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        x_center = int((xmin + xmax) / 2.)
        y_center = int((ymin + ymax) / 2.)
        width = xmax - xmin
        height = ymax - ymin
        choice = random.random()
        if choice < 0.5:
            # shrink or expand
            ratio = (random.random() * 3 + 1) / 2.
            height = height * ratio
            ratio = (random.random() * 3 + 1) / 2.
            width = width * ratio
        else:
            move_x = width * (random.random() * 4 - 2)
            move_y = height * (random.random() * 4 - 2)
            x_center += move_x
            y_center += move_y
        xmin = int(x_center - width / 2)
        xmax = int(x_center + width / 2)
        ymin = int(y_center - height / 2)
        ymax = int(y_center + height / 2)
        return [xmin, ymin, xmax, ymax]

    def unclip(self, box, expand_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * expand_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score(self, bitmap, box):
        """
        naive version of box score computation,
        only for helping principle understand.
        """
        mask = np.zeros_like(bitmap, dtype=np.uint8)
        cv2.fillPoly(mask, box.reshape(1, 4, 2).astype(np.int32), 1)
        return cv2.mean(bitmap, mask)[0]

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, 4, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]


def make_seg_postprocessor(config, is_train):
    top_n = config.MODEL.SEG.TOP_N_TRAIN
    if not is_train:
        top_n = config.MODEL.SEG.TOP_N_TEST

    binary_thresh = config.MODEL.SEG.BINARY_THRESH
    box_thresh = config.MODEL.SEG.BOX_THRESH
    min_size = config.MODEL.SEG.MIN_SIZE
    box_selector = SEGPostProcessor(
        top_n=top_n,
        binary_thresh=binary_thresh,
        box_thresh=box_thresh,
        min_size=min_size,
        cfg = config
    )
    return box_selector
