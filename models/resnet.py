'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlockConvKSize(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size, stride=1):
        super(BasicBlockConvKSize, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=kernel_size, stride=stride, padding=int(kernel_size/2), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size,
                               stride=1, padding=int(kernel_size/2), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlockSkipConvKSize(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, skip_kernel_size, conv_kernel_size, stride=1):
        super(BasicBlockSkipConvKSize, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=conv_kernel_size, stride=stride, padding=int(conv_kernel_size/2), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=conv_kernel_size,
                               stride=1, padding=int(conv_kernel_size/2), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=skip_kernel_size, stride=stride, padding=int(skip_kernel_size/2), bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNetThreeLayer1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetThreeLayer1, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNetThreeLayer2(nn.Module):
    def __init__(self, block, kernel_size, num_blocks, num_classes=10):
        super(ResNetThreeLayer2, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=kernel_size,
                               stride=1, padding=int(kernel_size/2), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, kernel_size,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, kernel_size,  num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, kernel_size,  num_blocks[2], stride=2)
        self.linear = nn.Linear(128*block.expansion, num_classes)

    def _make_layer(self, block, planes, kernel_size, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNetThreeLayer3(nn.Module):
    def __init__(self, block, skip_kernel_size, conv_kernel_size, num_blocks, num_classes=10):
        super(ResNetThreeLayer3, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=conv_kernel_size,
                               stride=1, padding=int(conv_kernel_size/2), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, skip_kernel_size, conv_kernel_size, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, skip_kernel_size, conv_kernel_size, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, skip_kernel_size, conv_kernel_size, num_blocks[2], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, skip_kernel_size, conv_kernel_size, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, skip_kernel_size, conv_kernel_size, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNetFourLayer1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetFourLayer1, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNetFourLayer2(nn.Module):
    def __init__(self, block, kernel_size, num_blocks, num_classes=10):
        super(ResNetFourLayer2, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=kernel_size,
                               stride=1, padding=int(kernel_size/2), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, kernel_size, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, kernel_size, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, kernel_size, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, kernel_size, num_blocks[3], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, kernel_size,  num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNetFourLayer3(nn.Module):
    def __init__(self, block, skip_kernel_size, conv_kernel_size, num_blocks, num_classes=10):
        super(ResNetFourLayer3, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=conv_kernel_size,
                               stride=1, padding=int(conv_kernel_size/2), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, skip_kernel_size, conv_kernel_size, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, skip_kernel_size, conv_kernel_size, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, skip_kernel_size, conv_kernel_size, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 64, skip_kernel_size, conv_kernel_size, num_blocks[3], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, skip_kernel_size, conv_kernel_size,  num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, skip_kernel_size, conv_kernel_size, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet_three_layer_1():
    return ResNetThreeLayer1(BasicBlock, [2, 2, 2])

def ResNet_three_layer_2():
    return ResNetThreeLayer1(BasicBlock, [1, 1, 1])

def ResNet_three_layer_3():
    return ResNetThreeLayer1(BasicBlock, [2, 1, 1])

def ResNet_three_layer_4():
    return ResNetThreeLayer1(BasicBlock, [1, 2, 1])

def ResNet_three_layer_5():
    return ResNetThreeLayer1(BasicBlock, [1, 1, 2])

def ResNet_three_layer_6():
    return ResNetThreeLayer1(BasicBlock, [2, 2, 1])

def ResNet_three_layer_7():
    return ResNetThreeLayer1(BasicBlock, [2, 1, 2])

def ResNet_three_layer_8():
    return ResNetThreeLayer1(BasicBlock, [1, 2, 2])

def ResNet_three_layer_9():
    return ResNetThreeLayer1(BasicBlock, [2, 2, 3])

def ResNet_three_layer_10():
    return ResNetThreeLayer1(BasicBlock, [1, 2, 3])

def ResNet_three_layer_11():
    return ResNetThreeLayer1(BasicBlock, [2, 1, 3])

def ResNet_three_layer_12():
    return ResNetThreeLayer1(BasicBlock, [2, 1, 4])

def ResNet_three_layer_13():
    return ResNetThreeLayer2(BasicBlockConvKSize, 3, [2, 2, 2])

def ResNet_three_layer_14():
    return ResNetThreeLayer2(BasicBlockConvKSize, 5, [2, 2, 2])

def ResNet_three_layer_15():
    return ResNetThreeLayer3(BasicBlockSkipConvKSize, 3, 3, [2, 2, 3])

def ResNet_three_layer_16():
    return ResNetThreeLayer3(BasicBlockSkipConvKSize, 5, 3, [2, 2, 3])

def ResNet_four_layer_1():
    return ResNetFourLayer1(BasicBlock, [1, 1, 1, 1])

def ResNet_four_layer_2():
    return ResNetFourLayer1(BasicBlock, [2, 1, 1, 1])

def ResNet_four_layer_3():
    return ResNetFourLayer2(BasicBlockConvKSize, 3, [2, 2, 2, 2])

def ResNet_four_layer_4():
    return ResNetFourLayer2(BasicBlockConvKSize, 3, [2, 2, 2, 3])

def ResNet_four_layer_5():
    return ResNetFourLayer2(BasicBlockConvKSize, 3, [2, 2, 3, 3])

def ResNet_four_layer_6():
    return ResNetFourLayer2(BasicBlockConvKSize, 3, [2, 3, 3, 3])

def ResNet_four_layer_7():
    return ResNetFourLayer2(BasicBlockConvKSize, 3, [3, 3, 3, 3])

def ResNet_four_layer_8():
    return ResNetFourLayer3(BasicBlockSkipConvKSize, 1, 3, [2, 2, 2, 2])

def ResNet_four_layer_9():
    return ResNetFourLayer3(BasicBlockSkipConvKSize, 3, 3, [2, 2, 2, 2])

def ResNet_four_layer_10():
    return ResNetFourLayer3(BasicBlockSkipConvKSize, 1, 5, [2, 2, 2, 2])

def ResNet_four_layer_11():
    return ResNetFourLayer3(BasicBlockSkipConvKSize, 3, 5, [2, 2, 2, 2])

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
