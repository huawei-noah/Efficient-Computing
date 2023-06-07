import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable



def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1):
        super(ResNet, self).__init__()
        self.in_planes = 2

        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2)
        self.layer1 = self._make_layer(block, 4, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 8, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 16, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 32, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, 64, num_blocks[4], stride=2)
        self.linear = nn.Linear(64, num_classes)
        
        self.in_planes = 32
        self.layer6 = self._make_layer(block, 64, 3, stride=2)
        self.linear1 = nn.Linear(64, 1)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, reg=False):
        #out = F.avg_pool2d(x, [10,2])
        out = F.relu(self.bn1(self.conv1(x)))
        #out = F.avg_pool2d(out, 10)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if reg:
            out1 = self.layer6(out)
            out1 = F.avg_pool2d(out1, [out1.size()[2],out1.size()[3]])
            out1 = out1.view(out1.size(0), -1)
            out1 = self.linear1(out1)
            
            out = self.layer5(out)
            out = F.avg_pool2d(out, [out.size()[2],out.size()[3]])
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out, out1
        out = self.layer5(out)
        out = F.avg_pool2d(out, [out.size()[2],out.size()[3]])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [1, 1, 1, 1, 1])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()