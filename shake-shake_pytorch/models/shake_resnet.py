# -*- coding: utf-8 -*-

import math

import torch.nn as nn
import torch.nn.functional as F

from models.shakeshake import ShakeShake
from models.shakeshake import Shortcut

from . import polynomial

class ShakeBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, use_shakeshake=False, act_type="relu"):
        super(ShakeBlock, self).__init__()
        self.use_shakeshake = use_shakeshake
        self.act = {
            "relu": lambda d: nn.ReLU(),
            "linkact": lambda d: polynomial.LinkActivation(2, d, n_degree=3),
            "regact": lambda d: polynomial.RegActivation(2, d, n_degree=3),
        }[act_type]
        self.equal_io = in_ch == out_ch
        self.shortcut = self.equal_io and None or Shortcut(in_ch, out_ch, stride=stride)

        self.branch1 = self._make_branch(in_ch, out_ch, stride)
        self.branch2 = self._make_branch(in_ch, out_ch, stride)

    def forward(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        
        if self.use_shakeshake:
            h = ShakeShake.apply(h1, h2, self.training)
        else:
            h = (h1 + h2) * 0.5
            
        h0 = x if self.equal_io else self.shortcut(x)
        return h + h0

    def _make_branch(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            self.act(in_ch),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            self.act(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch))


class ShakeResNet(nn.Module):

    def __init__(self, depth, w_base, label, use_shakeshake, act_type, input_channels):
        super(ShakeResNet, self).__init__()
        n_units = (depth - 2) / 6

        in_chs = [16, w_base, w_base * 2, w_base * 4]
        self.in_chs = in_chs

        self.c_in = nn.Conv2d(input_channels, in_chs[0], 3, padding=1)
        self.layer1 = self._make_layer(n_units, in_chs[0], in_chs[1])
        self.layer2 = self._make_layer(n_units, in_chs[1], in_chs[2], 2, use_shakeshake=use_shakeshake, act_type=act_type)
        self.layer3 = self._make_layer(n_units, in_chs[2], in_chs[3], 2, use_shakeshake=use_shakeshake, act_type=act_type)
        self.fc_out = nn.Linear(in_chs[3], label)

        # Initialize paramters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        h = self.c_in(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = F.relu(h)
        h = F.avg_pool2d(h, 8)
        h = h.view(-1, self.in_chs[3])
        h = self.fc_out(h)
        return h

    def _make_layer(self, n_units, in_ch, out_ch, stride=1, use_shakeshake=False, act_type="relu"):
        layers = []
        for i in range(int(n_units)):
            layers.append(ShakeBlock(in_ch, out_ch, stride=stride, use_shakeshake=use_shakeshake, act_type=act_type))
            in_ch, stride = out_ch, 1
        return nn.Sequential(*layers)
