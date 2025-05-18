import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import *
from pycm import *
from args import args_parser
import numpy as np
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import os

args = args_parser()
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

def validation(epoch, best_acc, val_loader, sd_net, net, optim_1, resume_path, criterion, mode, loss_coefficient):
	print('val at epoch {}'.format(epoch))
	
	sd_net.eval()
	net.eval()
	losses = AverageMeter()
	losses_local = AverageMeter()

	accuracies_1 = AverageMeter()
	accuracies_2 = AverageMeter()
	accuracies_3 = AverageMeter()
	accuracies_4 = AverageMeter()
	accuracies = AverageMeter()
	accuracies_total_ensemble = AverageMeter()

	if mode == 'va':
		with torch.no_grad():
			for i, (filenames, imgs_s1, labels) in enumerate(val_loader):
				with torch.no_grad():
					img_s1 = Variable(imgs_s1.to(device))     
					labels = Variable(labels.to(device))
					logits = sd_net(img_s1)
					loss = criterion(logits, labels)
					acc = accuracy(logits, labels)
				losses.update(loss.item(), logits.size(0))
				accuracies.update(acc, logits.size(0))

				if (i%50==0 and i!=0) or i+1==len(val_loader):
					print ('Validation:   Epoch[{}]:{}/{}    Loss:{:.4f}   Accu:{:.2f}%'.   \
							format(epoch, i, len(val_loader), float(losses.avg), float(accuracies.avg)*100))
		
		if accuracies.avg >= best_acc[0]:
			best_acc[0] = accuracies.avg
			save_file_path = resume_path
			states = {'state_dict': sd_net.state_dict(),
					'epoch':epoch,
					'optim_1': optim_1.state_dict(),
					'acc':best_acc[0]}
			torch.save(states, save_file_path)
			print ('Saved!')
		print ('curr_acc: {:.2f}%'.format(accuracies.avg*100))
		print ('best_acc: {:.2f}%'.format(best_acc[0]*100))
		
		
	
	elif mode == 'kd':
		for i, (filenames, imgs_s1, labels) in enumerate(val_loader):
			with torch.no_grad():
				imgs_s1 = Variable(imgs_s1.to(device))  
				labels = Variable(labels.to(device))

				logits, feat_map, feat_en = sd_net(imgs_s1)
				local_logits, weights, local_imgs, mask = net(imgs_s1, feat_map, feat_en)

				loss1 = torch.FloatTensor([0.]).to(device)
				loss2 = torch.FloatTensor([0.]).to(device)

				for index in range(len(logits)):
					loss1 += criterion(logits[index].to(device), labels)
				loss2 += criterion(local_logits, labels)

				Teacher_logits = logits[-1] + local_logits
				for index in range(len(logits)):
					loss1 += KD_loss(logits[index].to(device), Teacher_logits) * loss_coefficient
				loss2 += KD_loss(local_logits, Teacher_logits) * loss_coefficient

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

			if (i%50==0 and i!=0) or i+1==len(val_loader):
				print ('Validation:   Epoch[{}]:{}/{}    Loss_global:{:.4f}   Loss_local:{:.4f}   Acc1:{:.2f}%  Acc2:{:.2f}%'
					'   Acc3:{:.2f}%   Acc4:{:.2f}%   local_Acc:{:.2f}%  Acc_total_ensemble:{:.2f}%'.\
						format(epoch, i, len(val_loader), float(losses.avg), float(losses_local.avg), 
								float(accuracies_1.avg)*100, 
								float(accuracies_2.avg)*100, 
								float(accuracies_3.avg)*100, 
								float(accuracies_4.avg)*100, 
								float(accuracies.avg)*100, 
								float(accuracies_total_ensemble.avg)*100))
					
		print ('curr_acc1: {:.2f}%  curr_acc2: {:.2f}%  curr_acc3: {:.2f}%  curr_acc4: {:.2f}%  curr_local_Acc: {:.2f}%  curr_acc_total_ensemble: {:.2f}%'
		.format((accuracies_1.avg)*100, (accuracies_2.avg)*100, (accuracies_3.avg)*100, (accuracies_4.avg)*100, (accuracies.avg)*100, (accuracies_total_ensemble.avg)*100))
		
		if accuracies_1.avg >= best_acc[1]:
			best_acc[1] = accuracies_1.avg
		
		if accuracies_2.avg >= best_acc[2]:
			best_acc[2] = accuracies_2.avg

		if accuracies_3.avg >= best_acc[3]:
			best_acc[3] = accuracies_3.avg

		if accuracies_4.avg >= best_acc[4]:
			best_acc[4] = accuracies_4.avg
			
		if accuracies.avg >= best_acc[0]:
			best_acc[0] = accuracies.avg

		if accuracies_total_ensemble.avg >= best_acc[5]:
			best_acc[5] = accuracies_total_ensemble.avg
			save_file_path = resume_path
			states = {'Local_state_dict': net.state_dict(),
					'state_dict': sd_net.state_dict(),
					'epoch':epoch,
					'optim_1': optim_1.state_dict(),
					'acc':best_acc}
			torch.save(states, save_file_path)
			print ('Saved!')

		print ('best_acc1: {:.2f}%  best_acc2: {:.2f}%  best_acc3: {:.2f}%  best_acc4: {:.2f}%  best_local_Acc: {:.2f}%  best_total_acc_ensemble: {:.2f}%' 
				.format(best_acc[1]*100, best_acc[2]*100, best_acc[3]*100, best_acc[4]*100,  best_acc[0]*100, best_acc[5]*100))
		

	return best_acc, accuracies.avg
