#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator
from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor

from maskrcnn_benchmark.layers import Conv2d
import math

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

def keep_only_positive_boxes(boxes, batch_size_per_im):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        if len(inds) > batch_size_per_im:
            new_inds = inds[:batch_size_per_im]
            inds_mask[inds[batch_size_per_im:]] = 0
        else:
            new_inds = inds
        positive_boxes.append(boxes_per_image[new_inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


# TODO
def project_char_masks_on_boxes(
    segmentation_masks, segmentation_char_masks, proposals, discretization_size
):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    char_masks = []
    char_mask_weights = []
    decoder_targets = []
    word_targets = []
    M_H, M_W = discretization_size[0], discretization_size[1]
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )
    assert segmentation_char_masks.size == proposals.size, "{}, {}".format(
        segmentation_char_masks, proposals
    )
    # TODO put the proposals on the CPU, as the representation for the
    # masks is not efficient GPU-wise (possibly several small tensors for
    # representing a single instance mask)
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, segmentation_char_mask, proposal in zip(
        segmentation_masks, segmentation_char_masks, proposals
    ):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation,
        # instead of the list representation that was used
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M_W, M_H))
        mask = scaled_mask.convert(mode="mask")
        masks.append(mask)
        cropped_char_mask = segmentation_char_mask.crop(proposal)
        scaled_char_mask = cropped_char_mask.resize((M_W, M_H))
        char_mask, char_mask_weight, decoder_target, word_target = scaled_char_mask.convert(
            mode="seq_char_mask"
        )
        char_masks.append(char_mask)
        char_mask_weights.append(char_mask_weight)
        decoder_targets.append(decoder_target)
        word_targets.append(word_target)
    if len(masks) == 0:
        return (
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    return (
        torch.stack(masks, dim=0).to(device, dtype=torch.float32),
        torch.stack(char_masks, dim=0).to(device, dtype=torch.long),
        torch.stack(char_mask_weights, dim=0).to(device, dtype=torch.float32),
        torch.stack(decoder_targets, dim=0).to(device, dtype=torch.long),
        torch.stack(word_targets, dim=0).to(device, dtype=torch.long),
    )


class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg, proposal_matcher, discretization_size):
        super(ROIMaskHead, self).__init__()
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg)
        self.predictor = make_roi_mask_predictor(cfg)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)
        if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'ATTENTION':
            self.mask_attention = nn.Sequential(
                conv3x3_bn_relu(cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1] + 1, 32),
                conv3x3(32, 1),
                # Conv2d(cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1] + 1, 1, 1, 1, 0),
                nn.Sigmoid()
            )
            self.mask_attention.apply(self.weights_init)
            # for name, param in self.named_parameters():
            #     if "bias" in name:
            #         nn.init.constant_(param, 0)
            #     elif "weight" in name:
            #         # Caffe2 implementation uses MSRAFill, which in fact
            #         # corresponds to kaiming_normal_ in PyTorch
            #         nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
        if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'ATTENTION_DOWN':
            self.mask_attention = nn.Sequential(
                conv3x3_bn_relu(cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1] + 1, 32, stride=2),
                conv3x3(32, 1, stride=2),
                nn.Upsample(scale_factor=4, mode='nearest'),
                nn.Sigmoid()
            )
            self.mask_attention.apply(self.weights_init)

        if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'ATTENTION_CHANNEL':
            num_channel = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1] * 2
            self.channel_attention = nn.Sequential(
                nn.MaxPool2d(2),
                conv3x3_bn_relu(num_channel, num_channel, stride=2),
                conv3x3(num_channel, num_channel, stride=2),
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Sigmoid()
            )
            self.channel_attention.apply(self.weights_init)
        if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'ATTENTION_CHANNEL_SPLIT' or self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'ATTENTION_CHANNEL_SPLIT_BINARY':
            num_channel = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1] * 2
            self.channel_attention = nn.Sequential(
                nn.MaxPool2d(2),
                conv3x3_bn_relu(num_channel, int(num_channel / 4), stride=2),
                conv3x3(int(num_channel / 4), 2, stride=2),
                nn.AdaptiveAvgPool2d((1,1)),
                # nn.Sigmoid()
                nn.Softmax(dim=1)
            )
            self.channel_attention.apply(self.weights_init)
        if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'ATTENTION_CHANNEL_2':
            num_channel = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1] * 2
            self.channel_attention_2 = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Conv2d(
                    num_channel, num_channel, kernel_size=1, stride=1, padding=0
                ),
                nn.Conv2d(
                    num_channel, num_channel, kernel_size=1, stride=1, padding=0
                ),
                nn.Softmax(dim=1)
            )
            self.channel_attention_2.apply(self.weights_init)
        if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'ATTENTION_CHANNEL_TANH':
            feature_dim = 128
            num_channel = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1] * 2
            self.mask_pooler = nn.Sequential(
                nn.MaxPool2d(2),
                conv3x3_bn_relu(num_channel, num_channel, stride=2),
            )
            self.attn = nn.Linear(feature_dim, feature_dim)
            self.v = nn.Parameter(torch.rand(feature_dim))
            stdv = 1.0 / math.sqrt(self.v.size(0))
            self.v.data.normal_(mean=0, std=stdv)
            self.mask_pooler.apply(self.weights_init)
        if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'NEW_CAT':
            num_channel = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]
            self.enlarge_recepitve_field = nn.Sequential(
                nn.Conv2d(
                    2 * num_channel, num_channel, kernel_size=3, stride=1, padding=2, dilation=2
                ),
                nn.Conv2d(
                    num_channel, num_channel, kernel_size=3, stride=1, padding=2, dilation=2
                ),
            )
            self.enlarge_recepitve_field.apply(self.weights_init)
        if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'NEW_MASK':
            num_channel = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]
            self.new_mask = nn.Sequential(
                nn.Conv2d(
                    2 * num_channel, num_channel, kernel_size=3, stride=1, padding=2, dilation=2
                ),
                nn.Conv2d(
                    num_channel, 32, kernel_size=3, stride=1, padding=2, dilation=2
                ),
                nn.Conv2d(
                    32, 1, kernel_size=3, stride=1, padding=2, dilation=2
                ),
                nn.Sigmoid()
            )
            self.new_mask.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(1e-4)

    def step_function(self, x):
        return torch.reciprocal(1 + torch.exp(-50 * (x - 0.5)))

    def channel_attention_tanh(self, feature, mask):
        """
        :param hidden:
            previous hidden state of the decoder, in shape (B, hidden_size)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (H*W, B, hidden_size)
        :return
            attention energies in shape (B, H*W)
        """
        feature = feature.reshape((feature.shape[0], feature.shape[1], -1)) # (B, C, H*W)
        masks = mask.reshape((mask.shape[0], mask.shape[1], -1)).repeat(1, feature.shape[1], 1) # (B, C, H*W)
        fuse_feature = torch.cat([feature, masks], 2)
        energy = torch.tanh(self.attn(fuse_feature))  # (B, C, 2*H*W)->(B, C, 2*H*W)
        energy = energy.transpose(2, 1)  # (B, 2*H*W, C)
        v = self.v.repeat(feature.shape[0], 1).unsqueeze(
            1
        )  # (B, 1, 2*H*W)
        energy = torch.bmm(v, energy)  # (B, 1, C)
        energy = energy.squeeze(1)  # (B, C)
        return nn.functional.softmax(energy, dim=1).unsqueeze(2).unsqueeze(3) # normalize with softmax (B, C)

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        # match_quality_matrix = boxlist_polygon_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "masks", "char_masks"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        masks = []
        char_masks = []
        char_mask_weights = []
        decoder_targets = []
        word_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            segmentation_masks = matched_targets.get_field("masks")
            segmentation_masks = segmentation_masks[positive_inds]

            char_segmentation_masks = matched_targets.get_field("char_masks")
            char_segmentation_masks = char_segmentation_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            masks_per_image, char_masks_per_image, char_masks_weight_per_image, decoder_targets_per_image, word_targets_per_image = project_char_masks_on_boxes(
                segmentation_masks,
                char_segmentation_masks,
                positive_proposals,
                self.discretization_size,
            )

            masks.append(masks_per_image)
            char_masks.append(char_masks_per_image)
            char_mask_weights.append(char_masks_weight_per_image)
            decoder_targets.append(decoder_targets_per_image)
            word_targets.append(word_targets_per_image)

        return masks, char_masks, char_mask_weights, decoder_targets, word_targets
    
    def feature_mask(self, x, proposals):
        masks = []
        for proposal in proposals:
            segmentation_masks = proposal.get_field("masks")
            boxes = proposal.bbox.to(torch.device("cpu"))
            for segmentation_mask, box in zip(segmentation_masks, boxes):
                cropped_mask = segmentation_mask.crop(box)
                scaled_mask = cropped_mask.resize((self.cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION_W, self.cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION_H))
                mask = scaled_mask.convert(mode="mask")
                masks.append(mask)
        if len(masks) == 0:
            if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'CAT':
                x = cat([x, torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)], dim=1)
            if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'MIX' or 'ATTENTION_CHANNEL' in self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION:
                x = cat([x, x], dim=1)
            return x
        masks = torch.stack(masks, dim=0).to(x.device, dtype=torch.float32)
        if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'CAT':
            x = cat([x, masks.unsqueeze(1)], dim=1)
            return x
        if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'NEW_CAT':
            cat_x = cat([x, x * masks.unsqueeze(1)], dim=1)
            out_x = self.enlarge_recepitve_field(cat_x)
            return out_x
        if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'NEW_MASK':
            cat_x = cat([x, x * masks.unsqueeze(1)], dim=1)
            new_mask = self.new_mask(cat_x)
            out_x = x * new_mask
            return out_x
        if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'ATTENTION' or self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'ATTENTION_DOWN':
            x_cat = cat([x, masks.unsqueeze(1)], dim=1)
            attention = self.mask_attention(x_cat)
            x = x * attention
            return x
        if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'MIX':
            mask_x = x * masks.unsqueeze(1)
            cat_x = cat([x, mask_x], dim=1)
            return cat_x
        if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'ATTENTION_CHANNEL':
            mask_x = x * masks.unsqueeze(1)
            cat_x = cat([x, mask_x], dim=1)
            channel_attention = self.channel_attention(cat_x)
            attentioned_x = cat_x * channel_attention
            return attentioned_x
        if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'ATTENTION_CHANNEL_2':
            mask_x = x * masks.unsqueeze(1)
            cat_x = cat([x, mask_x], dim=1)
            channel_attention = self.channel_attention_2(cat_x)
            # print(channel_attention[0, :, 0, 0])
            attentioned_x = cat_x * channel_attention
            return attentioned_x
        if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'ATTENTION_CHANNEL_SPLIT':
            mask_x = x * masks.unsqueeze(1)
            cat_x = cat([x, mask_x], dim=1)
            channel_attention = self.channel_attention(cat_x)
            print(channel_attention[0, :, 0, 0])
            attentioned_x = cat([x * channel_attention[:, 0:1, :, :], mask_x * channel_attention[:, 1:, :, :]], dim=1)
            return attentioned_x
        if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'ATTENTION_CHANNEL_SPLIT_BINARY':
            mask_x = x * masks.unsqueeze(1)
            cat_x = cat([x, mask_x], dim=1)
            channel_attention = self.step_function(self.channel_attention(cat_x))
            # print(channel_attention[:, :, 0, 0])
            attentioned_x = cat([x * channel_attention[:, 0:1, :, :], mask_x * channel_attention[:, 1:, :, :]], dim=1)
            # attentioned_x = cat([x * channel_attention[:, 1:, :, :], mask_x * channel_attention[:, 0:1, :, :]], dim=1)
            return attentioned_x
        if self.cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'ATTENTION_CHANNEL_TANH':
            mask_x = x * masks.unsqueeze(1)
            cat_x = cat([x, mask_x], dim=1)
            pooler_x = self.mask_pooler(cat_x)
            pooler_mask = nn.functional.interpolate(masks.unsqueeze(1), scale_factor=0.25, mode='bilinear')
            channel_attention = self.channel_attention_tanh(pooler_x, pooler_mask)
            attentioned_x = cat_x * channel_attention
            return attentioned_x
        soft_ratio = self.cfg.MODEL.ROI_MASK_HEAD.SOFT_MASKED_FEATURE_RATIO
        if soft_ratio > 0:
            if soft_ratio < 1.0:
                x = x * (soft_ratio + (1 - soft_ratio) * masks.unsqueeze(1))
            else:
                x = x * (1.0 + soft_ratio * masks.unsqueeze(1))
        else:
            x = x * masks.unsqueeze(1)
        return x

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(
                proposals, self.cfg.MODEL.ROI_MASK_HEAD.MASK_BATCH_SIZE_PER_IM
            )
            if all(len(proposal) == 0 for proposal in proposals):
                return None, None, None
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)
            if self.cfg.MODEL.ROI_MASK_HEAD.USE_MASKED_FEATURE:
                x = self.feature_mask(x, proposals)
        if self.training:
            mask_targets, char_mask_targets, char_mask_weights, \
                decoder_targets, word_targets = self.prepare_targets(
                    proposals, targets
                )
            decoder_targets = cat(decoder_targets, dim=0)
            word_targets = cat(word_targets, dim=0)

        # proposals_not_empty, targets_not = [], []
        # for proposal, target, mask_target, char_mask_target, char_mask_weight in zip(proposals, targets, mask_targets, char_mask_targets, char_mask_weights):
        #     if len(proposal_target[0]) > 0:
        #         proposals_not_empty.append(proposal)
        #         targets_not.append(proposal_target[1])
        # proposals = proposals_not_empty
        # targets = targets_not
        if self.cfg.MODEL.CHAR_MASK_ON:
            if self.cfg.SEQUENCE.SEQ_ON:
                if not self.training:
                    if x.numel() > 0:
                        mask_logits, char_mask_logits, seq_outputs, seq_scores, \
                            detailed_seq_scores = self.predictor(x)
                        result = self.post_processor(
                            mask_logits,
                            char_mask_logits,
                            proposals,
                            seq_outputs=seq_outputs,
                            seq_scores=seq_scores,
                            detailed_seq_scores=detailed_seq_scores,
                        )
                        return x, result, {}
                    else:
                        return None, None, {}
                mask_logits, char_mask_logits, seq_outputs = self.predictor(
                    x, decoder_targets=decoder_targets, word_targets=word_targets
                )
                loss_mask, loss_char_mask = self.loss_evaluator(
                    proposals,
                    mask_logits,
                    char_mask_logits,
                    mask_targets,
                    char_mask_targets,
                    char_mask_weights,
                )
                return (
                    x,
                    all_proposals,
                    dict(
                        loss_mask=loss_mask,
                        loss_char_mask=loss_char_mask,
                        loss_seq=seq_outputs,
                    ),
                )
            else:
                mask_logits, char_mask_logits = self.predictor(x)
                if not self.training:
                    result = self.post_processor(
                        mask_logits, char_mask_logits, proposals
                    )
                    return x, result, {}
                loss_mask, loss_char_mask = self.loss_evaluator(
                    proposals,
                    mask_logits,
                    char_mask_logits,
                    mask_targets,
                    char_mask_targets,
                    char_mask_weights,
                )
                return (
                    x,
                    all_proposals,
                    dict(loss_mask=loss_mask, loss_char_mask=loss_char_mask),
                )
        else:
            if self.cfg.SEQUENCE.SEQ_ON:
                if self.cfg.MODEL.MASK_ON:
                    if not self.training:
                        if x.numel() > 0:
                            mask_logits, seq_outputs, seq_scores, \
                                detailed_seq_scores = self.predictor(x)
                            result = self.post_processor(
                                mask_logits,
                                None,
                                proposals,
                                seq_outputs=seq_outputs,
                                seq_scores=seq_scores,
                                detailed_seq_scores=detailed_seq_scores,
                            )
                            return x, result, {}
                        else:
                            return None, None, {}
                    mask_logits, seq_outputs = self.predictor(
                        x, decoder_targets=decoder_targets, word_targets=word_targets
                    )
                    loss_mask = self.loss_evaluator(
                        proposals,
                        mask_logits,
                        mask_targets,
                    )
                    return (
                        x,
                        all_proposals,
                        dict(
                            loss_mask=loss_mask,
                            loss_seq=seq_outputs,
                        ),
                    )
                else:
                    if not self.training:
                        if x.numel() > 0:
                            _, seq_outputs, seq_scores, \
                                detailed_seq_scores = self.predictor(x)
                            result = self.post_processor(
                                None,
                                None,
                                proposals,
                                seq_outputs=seq_outputs,
                                seq_scores=seq_scores,
                                detailed_seq_scores=detailed_seq_scores,
                            )
                            return x, result, {}
                        else:
                            return None, None, {}
                    _, seq_outputs = self.predictor(
                        x, decoder_targets=decoder_targets, word_targets=word_targets
                    )
                    return (
                        x,
                        all_proposals,
                        dict(
                            loss_seq=seq_outputs,
                        ),
                    )
            else:
                mask_logits = self.predictor(x)
                if not self.training:
                    result = self.post_processor(mask_logits, proposals)
                    return x, result, {}
                loss_mask = self.loss_evaluator(proposals, mask_logits, targets)
                return x, all_proposals, dict(loss_mask=loss_mask)


def build_roi_mask_head(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )
    return ROIMaskHead(
        cfg,
        matcher,
        (cfg.MODEL.ROI_MASK_HEAD.RESOLUTION_H, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION_W),
    )
