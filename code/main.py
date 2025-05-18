import os
import torch
import torch.nn as nn
from args import args_parser
from train import train
from val import validation
from datasets import load_datasets
from collections import OrderedDict
from models.MV2_model import *
from time import time
import numpy as np


args = args_parser()
best_acc = [0, 0, 0, 0, 0, 0]

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
    resume_path = args.resume_path.replace('dataset', args.dataset)  \
                                .replace('ratio', args.splits_ratio)   \
                                .replace('arch', args.arch)   \
                                .replace('local', args.local_arch)   \
                                .replace('mode', args.mode)   

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

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')    

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    params = list(sd_net.get_parameters()) + list(net.parameters())
    optim_1 = torch.optim.Adam(params, lr=args.lr)
    
    if os.path.exists(resume_path):
        if args.mode == "va":
            print ('Load model')
            resume = torch.load(resume_path, map_location= args.device)
            sd_net.load_state_dict(resume['state_dict'], strict=False)
            optim_1.load_state_dict(resume['optim_1'])
            
            sche_1 = torch.optim.lr_scheduler.StepLR(optim_1, step_size=args.step_size, last_epoch = resume['epoch'])
            args.start_epoch = resume['epoch'] + 1
            best_acc[0] = resume['acc']
            print('current epoch: ', resume['epoch'])
            print('Restored epoch: ', args.start_epoch)
            # print("sche_1 last_epoch: ", sche_1.last_epoch)
            print('current learning rate: ', optim_1.param_groups[0]['lr'])
            print('current best_acc: {:.2f}%' .format(best_acc[0]*100))
        if args.mode == "kd":
            print ('Load model')
            resume = torch.load(resume_path, map_location= args.device)
            sd_net.load_state_dict(resume['state_dict'], strict=False)
            net.load_state_dict(resume['Local_state_dict'], strict=False)
            optim_1.load_state_dict(resume['optim_1'])
            args.start_epoch = resume['epoch'] + 1
            sche_1 = torch.optim.lr_scheduler.StepLR(optim_1, step_size=args.step_size, last_epoch=resume['epoch'])
            best_acc = resume['acc']

            print('optim_1 current learning rate: ', optim_1.param_groups[0]['lr'])
            print('current epoch: ', resume['epoch'])
            print('Restored epoch: ', args.start_epoch)
            # print("sche_1 last_epoch: ", sche_1.last_epoch)
            print ('current best_acc1: {:.2f}%  best_acc2: {:.2f}%  best_acc3: {:.2f}%  best_acc4: {:.2f}%  best_local_Acc: {:.2f}%  best_total_acc_ensemble: {:.2f}%' 
				.format(best_acc[1]*100, best_acc[2]*100, best_acc[3]*100, best_acc[4]*100, best_acc[0]*100, best_acc[5]*100))
            
    else:
        sche_1 = torch.optim.lr_scheduler.StepLR(optim_1, step_size=args.step_size)

    # train model
    train_time = 0
    val_time = 0
    all_time = 0
    for i in range(args.start_epoch, args.epochs):

        beg_time = time()
        train_acc = train(i, train_loader, sd_net, net, optim_1, criterion, args.mode, args.loss_coefficient)
        end_time = time()
        train_time = train_time + (end_time - beg_time)
        all_time = all_time + (end_time - beg_time)
        print ('training_time: ', train_time)

        beg_time = time()
        best_acc, val_acc = validation(i, best_acc, val_loader, sd_net, net, optim_1, resume_path, criterion, args.mode, args.loss_coefficient)
        end_time = time()
        val_time = val_time + (end_time - beg_time)
        all_time = all_time + (end_time - beg_time)
        print ('validation_time: ', val_time)
        print ('all_time: ', all_time)
        print ('')	

        sche_1.step()
