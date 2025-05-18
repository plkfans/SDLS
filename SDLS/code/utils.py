import cv2
import numpy as np
from args import args_parser
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
import os

args = args_parser()
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return n_correct_elems / batch_size

def KD_loss(outputs, targets):
	kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/args.temperature, dim=1),F.softmax(targets/args.temperature, dim=1)) * args.temperature * args.temperature
	return kd_loss

def visualize_images(imgs_s1, local_imgs_s1, heatmaps, epoch, filename, weights, save_path=None):
    """
    可视化原图像、热图以及叠加后的图像。
    
    参数:
        imgs_s1: 原始图像张量 (B, C, H, W)。
        local_imgs_s1: 局部图像张量 (B, C, H, W)。
        heatmaps: 热图张量 (B, H, W)。
        epoch: 当前训练的 epoch 索引。
        save_path: 保存路径。
        weights: 权重
        filename: 文件名前缀。
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
    # 定义反归一化操作（如果图像有归一化处理）
    unnormalize = transforms.Compose([
        transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1/0.229, 1/0.224, 1/0.225])
    ])

    # 反归一化原图像
    imgs_s1 = unnormalize(imgs_s1.cpu())
    local_imgs_s1 = unnormalize(local_imgs_s1.cpu())

    # 转换为 (H, W, C) 格式以便可视化
    imgs_s1 = imgs_s1.permute(0, 2, 3, 1).numpy()
    local_imgs_s1 = local_imgs_s1.permute(0, 2, 3, 1).numpy()
    heatmaps = heatmaps.cpu().numpy()  # (B, H, W)

    # 获取样本
    img = imgs_s1[0]
    local_img = local_imgs_s1[0]
    heatmap = heatmaps[0]

    # 分别保存每个子图
    def save_single_plot(image, title, suffix):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, f"{filename}_{suffix}_epoch{epoch}.png"))
        plt.close()

    # 保存原图像
    save_single_plot(img, f"Original Image {filename} (Epoch {epoch})", "original")

    # 保存局部图像
    save_single_plot(
    local_img,
    f"Local Image {filename} (Epoch {epoch})\nweights: [w_global: {weights[0]:.6f}, w_local: {weights[1]:.6f}]",
    "local"
	)

    # 保存热图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(heatmap.squeeze(0), cmap='jet')
    ax.set_title(f"Heatmap {filename} (Epoch {epoch})")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{filename}_heatmap_epoch{epoch}.png"))
    plt.close()

    # 保存热图叠加图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.imshow(heatmap.squeeze(0), cmap='jet', alpha=0.5)
    ax.set_title(f"Heatmap Overlay {filename} (Epoch {epoch})")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{filename}_overlay_epoch{epoch}.png"))
    plt.close()