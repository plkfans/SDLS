import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.MV2_model import *
import torchvision.transforms as transforms
from args import args_parser
from val import validation
from datasets import load_datasets
from utils import *
from PIL import Image
import cv2
import numpy as np
from time import time
import pdb
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

args = args_parser()
best_acc = [0, 0, 0, 0, 0, 0, 0]

def label2cls(root, dataset, class_path):
    f = open(os.path.join(root, class_path.replace('dataset', dataset)), 'r')
    line = f.readline()
    label2cls_list ={}   
    while line:
        line = line.strip('\n')
        cls, label = line.split(' ')
        label2cls_list[str(label)] = str(cls)
        line = f.readline()
    return label2cls_list


def plot_confusion_matrix(dataset, true_list, pred_list, label2cls_list, mode):
    save_path = './save_status'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    labels = []
    for key, value in label2cls_list.items():
        labels.append(value)
    tick_marks = np.float32(np.array(range(len(labels)))) + 0.5

    cm = confusion_matrix(true_list, pred_list)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    # # UCM and RSSCN7
    # fontsize_axis = 5.2
    # fontsize_prop = 4.2

    # AID and NWPU-RESISC45
    fontsize_axis = 4.2
    fontsize_prop = 2.53

    barsize = 5
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_norm[y_val][x_val]
        if c > 0.01:
            color="white" if c > 0.5 else "black"
            plt.text(x_val, y_val, '%0.2f'%(c,), color=color, fontsize=fontsize_prop, va='center', ha='center')

    plt.gca().set_xticks(tick_marks)
    plt.gca().set_yticks(tick_marks)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-', linewidth=0.3)

    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues) 
    xlocations = np.array(range(len(labels))) 
    plt.xticks(xlocations, labels, fontsize=fontsize_axis, rotation=270) 
    plt.yticks(xlocations, labels, fontsize=fontsize_axis) 

    cb = plt.colorbar(shrink=1.0) 
    cb.ax.tick_params(labelsize=barsize)

    plt.tight_layout()
    plt.savefig('./save_status/confusion_matrix_' + dataset + '_' + mode + '.pdf', format='pdf')

# attention
if __name__ == '__main__':  

    # load device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # load datasets feat
    train_list = args.train_list.replace('dataset', args.dataset)
    val_list = args.val_list.replace('dataset', args.dataset)
   
    
    train_loader, val_loader = load_datasets(args.data_dir, 
                                             train_list, 
                                             val_list, 
                                             args.mode, 
                                             args.batch_size, 
                                             args.img_size, 
                                             args.n_workers)

    # bulid model
    if args.dataset=='AID':
        n_classes = 30
    elif args.dataset=='UCMerced_LandUse':
        n_classes = 21
    elif args.dataset=='NWPU-RESISC45':
        n_classes = 45
    elif args.dataset=='RSSCN7':
        n_classes = 7

    sd_net = MV2Model(arch=args.arch,  
                    n_classes=n_classes, 
                    mode=args.mode).to(device)
                    
    if args.local_arch=='MobileNetV2':
        net = branch3_MobileNetV2(num_classes=n_classes).to(device)
    elif args.local_arch=='efficientnet_b3':
        net = Efficientnet(num_classes=n_classes).to(device)

    resume_path = args.resume_path.replace('dataset', args.dataset)  \
                                .replace('ratio', args.splits_ratio)   \
                                .replace('arch', args.arch)   \
                                .replace('local', args.local_arch)   \
                                .replace('mode', args.mode) 

    if os.path.exists(resume_path):
        if args.mode == "va":
            print ('Load model')
            resume = torch.load(resume_path, map_location= args.device)
            sd_net.load_state_dict(resume['state_dict'], strict=False)
            # optim_1.load_state_dict(resume['optim_1'])
            
            # sche_1 = torch.optim.lr_scheduler.StepLR(optim_1, step_size=args.step_size, last_epoch = resume['epoch'])
            # args.start_epoch = resume['epoch'] + 1
            best_acc[0] = resume['acc']
            # print('current epoch: ', resume['epoch'])
            # print('Restored epoch: ', args.start_epoch)
            # print("sche_1 last_epoch: ", sche_1.last_epoch)
            # print('current learning rate: ', optim_1.param_groups[0]['lr'])
            print('current best_acc: {:.2f}%' .format(best_acc[0]*100))
        if args.mode == "kd":
            print ('Load model')
            resume = torch.load(resume_path, map_location= args.device)
            sd_net.load_state_dict(resume['state_dict'], strict=False)
            net.load_state_dict(resume['Local_state_dict'], strict=False)
            # optim_1.load_state_dict(resume['optim_1'])
            # args.start_epoch = resume['epoch'] + 1
            # sche_1 = torch.optim.lr_scheduler.StepLR(optim_1, step_size=args.step_size, last_epoch=resume['epoch'])
            best_acc = resume['acc']

            # print('optim_1 current learning rate: ', optim_1.param_groups[0]['lr'])
            # print('current epoch: ', resume['epoch'])
            # print('Restored epoch: ', args.start_epoch)
            # print("sche_1 last_epoch: ", sche_1.last_epoch)
            print ('current best_acc1: {:.2f}%  best_acc2: {:.2f}%  best_acc3: {:.2f}%  best_acc4: {:.2f}%  best_local_Acc: {:.2f}%  best_total_acc_ensemble: {:.2f}%' 
				.format(best_acc[1]*100, best_acc[2]*100, best_acc[3]*100, best_acc[4]*100, best_acc[0]*100, best_acc[5]*100))
    
    sd_net = sd_net.eval()
    net = net.eval()

    label2cls_list = label2cls(args.data_dir, args.dataset, args.class_list)

    if args.mode == "va":
        cnt_total = 0
        pred_list = []
        label_list = []

        beg_time = time()
        with torch.no_grad():
            for filenames, imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                preds = sd_net(imgs)
                _, preds = torch.max(preds, dim=1)

                pred_list.extend(preds.cpu().numpy())
                label_list.extend(labels.cpu().numpy())

        
        plot_confusion_matrix(args.dataset, label_list, pred_list, label2cls_list, args.mode)
        end_time = time()
        print("Total time:", end_time - beg_time)

    if args.mode == "kd":
        cnt_total = 0
        pred_list = []
        label_list = []

        beg_time = time()
        
        for filenames, imgs, labels in val_loader:
            with torch.no_grad():
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits, feat_map, feat_en = sd_net(imgs)
                local_logits, _, _, _ = net(imgs, feat_map, feat_en)
                preds = logits[-1] + local_logits
                _, preds = torch.max(preds, dim=1)
                pred_list.extend(preds.cpu().numpy())
                label_list.extend(labels.cpu().numpy())

        plot_confusion_matrix(args.dataset, label_list, pred_list, label2cls_list, args.mode)
        end_time = time()
        print("Total time:", end_time - beg_time)

