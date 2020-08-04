# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from collections import OrderedDict

# from torch import nn

# from . import fpn as fpn_module
# from . import resnet


# def build_resnet_backbone(cfg):
#     body = resnet.ResNet(cfg)
#     model = nn.Sequential(OrderedDict([("body", body)]))
#     return model


# def build_resnet_fpn_backbone(cfg):
#     body = resnet.ResNet(cfg)
#     in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
#     out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
#     fpn = fpn_module.FPN(
#         in_channels_list=[
#             in_channels_stage2,
#             in_channels_stage2 * 2,
#             in_channels_stage2 * 4,
#             in_channels_stage2 * 8,
#         ],
#         out_channels=out_channels,
#         top_blocks=fpn_module.LastLevelMaxPool(),
#     )
#     model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
#     return model


# _BACKBONES = {"resnet": build_resnet_backbone, "resnet-fpn": build_resnet_fpn_backbone}


# def build_backbone(cfg):
#     assert cfg.MODEL.BACKBONE.CONV_BODY.startswith(
#         "R-"
#     ), "Only ResNet and ResNeXt models are currently implemented"
#     # Models using FPN end with "-FPN"
#     if cfg.MODEL.BACKBONE.CONV_BODY.endswith("-FPN"):
#         return build_resnet_fpn_backbone(cfg)
#     return build_resnet_backbone(cfg)


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
# from . import resnet


@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model

@registry.BACKBONES.register("R-18-FPN")
@registry.BACKBONES.register("R-34-FPN")
@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    if cfg.MODEL.RESNET34:
        from . import resnet34 as resnet
        body = resnet.ResNet(layers=cfg.MODEL.RESNETS.LAYERS)
    else:
        from . import resnet
        body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
