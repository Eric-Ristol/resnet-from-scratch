#ResNet-20 for CIFAR-10, built from scratch.
#
#Based on: "Deep Residual Learning for Image Recognition" (He et al., 2015)
#https://arxiv.org/abs/1512.03385
#
#--- The core idea ---
#A plain deep network tries to learn some mapping y = H(x) directly.
#A residual block instead learns F(x) = H(x) - x, and outputs:
#
#       y = F(x, {W_i}) + x
#
#The `+ x` is the "identity shortcut". If the optimal mapping is close to
#identity (very common for nearby layers), the block only has to push F(x)
#toward zero — much easier than learning identity from scratch with nonlinear
#layers. Empirically this lets us train nets that are 100+ layers deep without
#vanishing gradients hurting the optimisation.
#
#--- Architecture (CIFAR-10 variant from the paper, Table 6) ---
#      layer name   output size      ResNet-20
#      ----------   -----------      ---------
#      conv1        32 x 32 x 16     3x3, 16 filters, stride 1
#      conv2_x      32 x 32 x 16     3 x BasicBlock(16)
#      conv3_x      16 x 16 x 32     3 x BasicBlock(32)   (first block has stride 2)
#      conv4_x      8  x 8  x 64     3 x BasicBlock(64)   (first block has stride 2)
#      avgpool      1  x 1  x 64     global average pool
#      fc           10               fully-connected to 10 classes
#
#Total parameters: ~272K.
#
#Shortcut type: we use "option A" (zero-padded identity) when the block
#changes the number of channels. It's parameter-free, keeps the "from scratch"
#spirit, and matches what He et al. used in the original CIFAR experiments.

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    #Two 3x3 convs with batch norm and ReLU, plus the identity shortcut.
    #If `stride != 1` or in/out channels differ, we downsample the shortcut
    #with average pooling and zero-pad extra channels (option A).

    expansion = 1  #kept for compatibility with deeper bottleneck variants

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        #Remember these for the shortcut path.
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride       = stride

    def _shortcut(self, x):
        #Option A shortcut: downsample spatially with avg-pool if stride>1,
        #then zero-pad the channel dimension if we grew wider. No parameters.
        if self.stride != 1:
            x = F.avg_pool2d(x, kernel_size=1, stride=self.stride)
        if self.in_channels != self.out_channels:
            pad = self.out_channels - self.in_channels
            #pad format: (W_left, W_right, H_top, H_bottom, C_front, C_back)
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
        return x

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self._shortcut(x)
        return F.relu(out, inplace=True)


class ResNetCifar(nn.Module):
    #Configurable CIFAR-style ResNet. `n` is the number of BasicBlocks
    #per stage, so total layer count = 6n + 2 (the +2 accounts for the first
    #conv and the final FC). n=3 gives ResNet-20; n=5 gives ResNet-32.

    def __init__(self, n=3, num_classes=10):
        super().__init__()
        self.n = n

        #conv1: 3x3, 16 filters, stride 1. No bias because BN handles it.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)

        #Three stages, each with n BasicBlocks.
        self.stage1 = self._make_stage(in_ch=16, out_ch=16, blocks=n, stride=1)
        self.stage2 = self._make_stage(in_ch=16, out_ch=32, blocks=n, stride=2)
        self.stage3 = self._make_stage(in_ch=32, out_ch=64, blocks=n, stride=2)

        #Global average pool → FC.
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(64, num_classes)

        self._init_weights()

    def _make_stage(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        #He initialisation for conv layers, standard for ResNets.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)           #[B, 64, 1, 1]
        x = torch.flatten(x, 1)       #[B, 64]
        return self.fc(x)             #[B, num_classes]


def resnet20(num_classes=10):
    #Factory helper — ResNet-20 is the canonical small CIFAR variant.
    return ResNetCifar(n=3, num_classes=num_classes)


def resnet32(num_classes=10):
    #Included for easy experimentation once the training loop is working.
    return ResNetCifar(n=5, num_classes=num_classes)


def count_parameters(model):
    #Small utility — nice to print at train time and to check in tests.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
