import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import *
from args import args_parser
import csv
import os
from pycm import *
import torch.nn.functional as F


args = args_parser()
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

def train(epoch, train_loader, sd_net, net, optim_1, criterion, mode, loss_coefficient):
    print('train at epoch {}'.format(epoch))

    sd_net.train()
    net.train()
    losses = AverageMeter()
    losses_local = AverageMeter()

    accuracies_1 = AverageMeter()
    accuracies_2 = AverageMeter()
    accuracies_3 = AverageMeter()
    accuracies_4 = AverageMeter()
    accuracies = AverageMeter()
    accuracies_total_ensemble = AverageMeter()

    if mode == 'va':
        for i, (filenames, imgs_s1, labels) in enumerate(train_loader):
            imgs_s1 = Variable(imgs_s1.to(device))    
            labels = Variable(labels.to(device))
            logits = sd_net(imgs_s1)

            optim_1.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            optim_1.step()
            
            acc = accuracy(logits, labels)
            losses.update(loss.item(), logits.size(0))        
            accuracies.update(acc, logits.size(0))

            if (i%50==0 and i!=0) or i+1==len(train_loader):
                current_lr = optim_1.param_groups[0]['lr']
                print ('Train:  Epoch[{}]:{}/{}  Learning Rate: {}  Loss:{:.4f}   Accu:{:.2f}%'.\
                        format(epoch, i, len(train_loader), current_lr, float(losses.avg), float(accuracies.avg)*100))
                
    elif mode == 'kd':
        for i, (filenames, imgs_s1, labels) in enumerate(train_loader):
            imgs_s1 = Variable(imgs_s1.to(device))    
            labels = Variable(labels.to(device))

            
            logits, feat_map, feat_en = sd_net(imgs_s1)
            
            feat_map = feat_map.detach()
            local_logits, _, _, _ = net(imgs_s1, feat_map, feat_en)

            optim_1.zero_grad()

            loss1 = torch.FloatTensor([0.]).to(device)
            loss2 = torch.FloatTensor([0.]).to(device)

            for index in range(len(logits)):
                loss1 += criterion(logits[index], labels)
            loss2 += criterion(local_logits, labels)

            Teacher_logits = logits[-1].detach() + local_logits.detach()
            for index in range(len(logits)):
                loss1 += KD_loss(logits[index], Teacher_logits) * loss_coefficient
            loss2 += KD_loss(local_logits, Teacher_logits) * loss_coefficient

            loss = loss1 + loss2
            loss.backward()
            optim_1.step()

            # 打印评价指标
            acc1 = accuracy(logits[0], labels)
            acc2 = accuracy(logits[1], labels)
            acc3 = accuracy(logits[2], labels)
            acc4 = accuracy(logits[3], labels)
            acc_local = accuracy(local_logits, labels)
            acc_total_ensemble = accuracy(Teacher_logits, labels)
            

            losses.update(loss1.item(), logits[0].size(0))   
            losses_local.update(loss2.item(), logits[0].size(0))   

            accuracies_1.update(acc1, logits[0].size(0))
            accuracies_2.update(acc2, logits[0].size(0))
            accuracies_3.update(acc3, logits[0].size(0))
            accuracies_4.update(acc4, logits[0].size(0))
            accuracies.update(acc_local, logits[0].size(0))
            accuracies_total_ensemble.update(acc_total_ensemble, logits[0].size(0))

            if (i%50==0 and i!=0) or i+1==len(train_loader):
                # Get the learning rate from the optimizer
                current_lr_1 = optim_1.param_groups[0]['lr']
                
                print ('Train:   Epoch[{}]:{}/{}   Loss_global:{:.4f}   Loss_local:{:.4f}   LR:{}  Acc1:{:.2f}%  Acc2:{:.2f}%'
                        '   Acc3:{:.2f}%   Acc4:{:.2f}%   local_Acc:{:.2f}%   Acc_total_ensemble:{:.2f}%' .\
                        format(epoch, i, len(train_loader), float(losses.avg), float(losses_local.avg),  current_lr_1, 
                            float(accuracies_1.avg)*100,
                            float(accuracies_2.avg)*100, 
                            float(accuracies_3.avg)*100, 
                            float(accuracies_4.avg)*100, 
                            float(accuracies.avg)*100, 
                            float(accuracies_total_ensemble.avg)*100))
                

    return accuracies.avg


 