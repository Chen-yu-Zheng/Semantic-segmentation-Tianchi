import torch
from torch import nn
import torch.nn.functional as F

import models.backbones.resnet as resnets

from utils.loss import DiceLoss

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes >= 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        self.classes = classes

        if layers == 50:
            resnet = resnets.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = resnets.resnet101(pretrained=pretrained)
        else:
            resnet = resnets.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # layer3 and layer4 don't change feature size.
        # use padding to fix dilation 
        # make conv which stride = 2 to 1
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2

        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, self.classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, self.classes, kernel_size=1)
            )

    def forward(self, x, y):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        
        x = x.squeeze()

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            
            aux = aux.squeeze()
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            dice_loss = DiceLoss()(x, y)

            if self.classes > 1:
                #返回最大值对应的索引值
                return x.max(1)[1], main_loss, aux_loss, dice_loss
            else:
                return (x >= 0), main_loss, aux_loss, dice_loss

        else:
            #不是训练时要另外算loss，即没有辅助Loss
            main_loss = self.criterion(x, y)
            dice_loss = DiceLoss()(x, y)

            if self.classes > 1:
                #返回最大值对应的索引值
                return x.max(1)[1], main_loss, dice_loss
            else:
                #便于与之前一致进行最后结果的推理
                return x, main_loss, dice_loss


if __name__ == '__main__':
    input = torch.rand(4, 3, 257, 257).cuda()
    model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=1, zoom_factor=8, use_ppm=True, pretrained=True).cuda()
    model.eval()
    output = model(input)
    print('PSPNet', output.shape)