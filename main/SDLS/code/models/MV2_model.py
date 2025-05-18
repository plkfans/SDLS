import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from . import *
from utils import *
import math
from args import args_parser
import numpy as np
import cv2
import random
import torchvision.models as models
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

args = args_parser()
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

def GroupConv(channel_in, channel_out, groups):
    return nn.Sequential(
        nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=1, groups=groups),
        nn.BatchNorm2d(channel_out),
        nn.ReLU6(inplace=True),
        )

def DWConv(channel_in, channel_out):
    return nn.Sequential(
        nn.Conv2d(channel_in, channel_in, kernel_size=3, stride=1, padding=1, groups=channel_in),
        nn.Conv2d(channel_in, channel_out, kernel_size=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU6(inplace=True),
        )

def branchBottleNeck(channel_in, channel_out):

    return nn.Sequential(
        nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU6(inplace=True),
        )

def channelBottleNeck(channel_in, channel_out):

    return nn.Sequential(
        nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        )

# 自蒸馏分支网络
class branch1_MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, channel_in = 64, channel_out = 24):
        super(branch1_MobileNetV2, self).__init__()
        # 加载预训练的 MobileNetV2 模型
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        
        self.initial_layer = mobilenet.features[0]
        self.block1 = mobilenet.features[1:4]  # 第 1 到 3 个 block
        self.block2 = mobilenet.features[4:7]  # 第 4 到 6 个 block
        self.block3 = mobilenet.features[7:14]   
        self.block4 = mobilenet.features[14:]   # 其余部分

        self.BottleNeck = branchBottleNeck(channel_in, channel_out)
        self.DWConv = DWConv(channel_in, channel_out)
        self.GroupConv = GroupConv(channel_in, channel_out, groups=8)
        self.Sigmoid = nn.Sigmoid()

        # 修改最后的全连接层，输出类别数为 num_classes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = mobilenet.classifier  # 最后的全连接层部分
        self.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)

    def forward(self, x, fea):
        # 保存每层的特征
        x = self.initial_layer(x)

        # 分阶段计算
        x1 = self.block1(x)
        mask1 = self.BottleNeck(fea)
        mask2 = self.DWConv(fea)
        mask3 = self.GroupConv(fea)
        mask = (mask1 + mask2 + mask3) 
        mask = self.Sigmoid(mask)

        x1 = mask * x1
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        # 返回最终输出和中间特征
        return x, x4
    
class branch2_MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, channel_in = 64, channel_out = 24):
        super(branch2_MobileNetV2, self).__init__()
        # 加载预训练的 MobileNetV2 模型
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        
        self.initial_layer = mobilenet.features[0]
        self.block1 = mobilenet.features[1:4]  # 第 1 到 3 个 block
        self.block2 = mobilenet.features[4:7]  # 第 4 到 6 个 block
        self.block3 = mobilenet.features[7:14]   
        self.block4 = mobilenet.features[14:]   # 其余部分

        self.BottleNeck = branchBottleNeck(channel_in, channel_out)
        self.DWConv = DWConv(channel_in, channel_out)
        self.GroupConv = GroupConv(channel_in, channel_out, groups=8)
        self.Sigmoid = nn.Sigmoid()

        # 修改最后的全连接层，输出类别数为 num_classes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = mobilenet.classifier  # 最后的全连接层部分
        self.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)

    def forward(self, x, fea):
        # 保存每层的特征
        x = self.initial_layer(x)

        # 分阶段计算
        x1 = self.block1(x)
        x2 = self.block2(x1)

        mask1 = self.BottleNeck(fea)
        mask2 = self.DWConv(fea)
        mask3 = self.GroupConv(fea)
        mask = (mask1 + mask2 + mask3) 
        mask = self.Sigmoid(mask)

        x2 = mask * x2
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        # 返回最终输出和中间特征
        return x, x4

# 局部网络
class branch3_MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(branch3_MobileNetV2, self).__init__()
        # 加载预训练的 MobileNetV2 模型
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        
        self.initial_layer = mobilenet.features[0]
        self.block1 = mobilenet.features[1:4]  # 第 1 到 3 个 block
        self.block2 = mobilenet.features[4:7]  # 第 4 到 6 个 block
        self.block3 = mobilenet.features[7:14]   
        self.block4 = mobilenet.features[14:]   # 其余部分

        self.weights = nn.Parameter(torch.tensor([0.2, 0.8]))

        # 修改最后的全连接层，输出类别数为 num_classes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = mobilenet.classifier  # 最后的全连接层部分
        self.classifier[1] = nn.Linear(in_features=1280*2, out_features=num_classes)

    def forward(self, x, fea, T_fea):

        mask = fea.mean(dim=1, keepdim=True)
        att_min = mask.amin(dim=(2, 3), keepdim=True)
        att_max = mask.amax(dim=(2, 3), keepdim=True)
        mask = (mask - att_min) / (att_max - att_min + 1e-6)
        mask = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=True)
        mask = mask ** 4.0
        local_imgs = self.weights[0] * x + self.weights[1] * (x * mask)

        x = self.initial_layer(local_imgs)

        # 分阶段计算
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x = self.avgpool(x4)
        fea = torch.flatten(x, 1)
        fea = torch.cat([T_fea, fea], dim=1)
        x = self.classifier(fea)
        
        # 返回最终输出和中间特征
        return x, self.weights, local_imgs, mask

class Efficientnet(nn.Module):
    def __init__(self, num_classes=1000):
        super(Efficientnet, self).__init__()
        # 加载预训练的 MobileNetV2 模型
        efficientnet = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        
        self.features_1 = efficientnet.features[0:3]
        self.features_2 = efficientnet.features[3]
        self.features_3 = efficientnet.features[4:]
        self.weights = nn.Parameter(torch.tensor([0.2, 0.8]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = efficientnet.classifier
        self.classifier[1] = nn.Linear(in_features=1536+1280, out_features=num_classes)

    def get_features(self, x):
        
        x1 = self.features_1(x)
        x2 = self.features_2(x1)
        x3 = self.features_3(x2)
        
        return [x1,x2,x3]

    def forward(self, x, fea, T_fea):

        mask = fea.mean(dim=1, keepdim=True)
        att_min = mask.amin(dim=(2, 3), keepdim=True)
        att_max = mask.amax(dim=(2, 3), keepdim=True)
        mask = (mask - att_min) / (att_max - att_min + 1e-6)
        mask = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=True)
        mask = mask ** 4.0
        local_imgs = self.weights[0] * x + self.weights[1] * (x * mask)

        x1 = self.features_1(local_imgs)
        x2 = self.features_2(x1)
        x3 = self.features_3(x2)
        x = self.avgpool(x3)
        fea = torch.flatten(x, 1)
        fea = torch.cat([T_fea, fea], dim=1)
        x = self.classifier(fea)
        
        return x, self.weights, local_imgs, mask

class MV2Model(nn.Module):
    def __init__(self, arch, n_classes, mode):
        self.mode = mode
        super(MV2Model, self).__init__()
        if arch == 'resnet18':
            self.feature_dim = 512
            self.channel_in = [64, 128]
            self.base = resnet18(pretrained=True)
        elif arch == 'shufflenet_v2_x1_5':
            self.feature_dim = 1024
            self.channel_in = [24, 176]
            self.base = shufflenet_v2_x1_5(pretrained=True)
        elif arch == 'wide_resnet50_2':
            self.feature_dim = 2048
            self.channel_in = [256, 512]
            self.base = wide_resnet50_2(pretrained=True)
        elif arch == 'efficientnet_b3':
            self.feature_dim = 1536
            self.channel_in = [32, 48]
            self.base = Efficientnet(num_classes=n_classes)

        self.scala1 = branch1_MobileNetV2(num_classes=n_classes, channel_in=self.channel_in[0], channel_out=24)
        self.scala2 = branch2_MobileNetV2(num_classes=n_classes, channel_in=self.channel_in[1], channel_out=32)
        self.branchBottleNeck = channelBottleNeck(self.feature_dim, 1280)

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.feature_dim, n_classes)
        self.fc_en = nn.Linear(1280, n_classes)

    def get_parameters(self):

        if self.mode == 'va':
            for param in self.scala1.parameters():
                param.requires_grad = False
            for param in self.scala2.parameters():
                param.requires_grad = False
            for param in self.branchBottleNeck.parameters():
                param.requires_grad = False
            for param in self.fc_en.parameters():
                param.requires_grad = False
            params = list(self.base.parameters()) + list(self.fc.parameters())

        elif self.mode == 'kd':
            params = list(self.base.parameters()) + \
                     list(self.scala1.parameters()) + \
                     list(self.scala2.parameters()) + \
                     list(self.branchBottleNeck.parameters()) + \
                     list(self.fc.parameters()) + \
                     list(self.fc_en.parameters())
        return params

    def forward(self, img_s1):
        img_b = img_s1.size(0)
        
        if self.mode == 'va':
            feat_maps = self.base.get_features(img_s1)
            logits_s4 = self.fc(self.pooling(feat_maps[-1]).view(img_s1.size(0), -1))
            logits = logits_s4
        
            return logits

        elif self.mode == 'kd':
            feat_maps = self.base.get_features(img_s1)
            logits_s1, feat_maps_s1 = self.scala1(img_s1, feat_maps[0])
            logits_s2, feat_maps_s2 = self.scala2(img_s1, feat_maps[1])
            fea3 = self.pooling(feat_maps[-1]).view(img_b, -1)
            logits_s3 = self.fc(fea3)

            feat_maps_s3 = self.branchBottleNeck(feat_maps[-1])
            feat_en = feat_maps_s1 + feat_maps_s2 + feat_maps_s3
            feat_map = feat_en
            feat_en = self.pooling(feat_en).view(img_b, -1)
            logits_en = self.fc_en(feat_en)

            logits = [logits_s1, logits_s2, logits_s3, logits_en]
            
        return logits, feat_map, feat_en


            