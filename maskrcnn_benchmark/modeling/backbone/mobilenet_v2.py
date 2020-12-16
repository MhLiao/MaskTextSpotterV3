import torch.nn as nn
import math
import torch
from collections import namedtuple
from maskrcnn_benchmark.layers import FrozenBatchNorm2d
# s0 = top layer idx
# name = sub op name
# s1 = sub layer idx
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #

def conv_bn(inp, oup, stride, norm_func):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        # nn.BatchNorm2d(oup),
        norm_func(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup, norm_func):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        # nn.BatchNorm2d(oup),
        norm_func(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_func):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                # nn.BatchNorm2d(hidden_dim),
                norm_func(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                # nn.BatchNorm2d(oup),
                norm_func(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                # nn.BatchNorm2d(hidden_dim),
                norm_func(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                # nn.BatchNorm2d(hidden_dim),
                norm_func(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                # nn.BatchNorm2d(oup),
                norm_func(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon=0.1):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


featuremap_indexes = [
    GraphPath(3, 'conv', 3), # stride = 4     chn = 144
    GraphPath(6, 'conv', 3), # stride = 8     chn = 192
    GraphPath(13, 'conv', 3), # stride = 16   chn = 576
    GraphPath(17, 'conv', 3), # stride = 32   chn = 1280
]
class MobileNetV2(nn.Module):
    def __init__(self, cfg, n_class=1000, input_size=224, width_mult=1., smooth=False):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],  # stride = 4
            [6, 32, 3, 2],  # stride = 8
            [6, 64, 4, 2],  # stride = 16
            [6, 96, 3, 1],
            [6, 160, 3, 2], # stride = 32
            [6, 320, 1, 1],
        ]
        norm_func = nn.BatchNorm2d if cfg.MODEL.MOBILENET.FROZEN_BN == False else FrozenBatchNorm2d

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, norm_func)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, norm_func=norm_func))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, norm_func=norm_func))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel, norm_func))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )
        # self.criterion = CrossEntropyLabelSmooth(n_class)

        self.criterion = nn.CrossEntropyLoss() if not smooth else CrossEntropyLabelSmooth(n_class)
        self._initialize_weights()

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for layer_index in range(freeze_at):
            for p in self.features[layer_index].parameters():
                print(p.size())
                p.requires_grad = False

    # def forward(self, x, target):
    #     x = self.features(x)
    #     x = x.mean(3).mean(2)
    #     logits = self.classifier(x)
    #     loss = self.criterion(logits, target)
    #     return logits, loss.unsqueeze(0)

    def forward(self, x):
        outputs = []
        fm_idx = 0
        for index, layer in enumerate(self.features):
            # print(index, fm_idx)
            if fm_idx < len(featuremap_indexes) and index == featuremap_indexes[fm_idx].s0:
                sub = getattr(layer, featuremap_indexes[fm_idx].name)
                for layer in sub[:featuremap_indexes[fm_idx].s1]:
                    x = layer(x)
                y = x
                for layer in sub[featuremap_indexes[fm_idx].s1:]:
                    x = layer(x)
                           
                fm_idx+=1
                # print(y.size())
                outputs.append(y)
            else:
                x =layer(x)
        # print(x.size())
        # # add last layer
        # outputs.append(x)
        return outputs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
