import argparse

def args_parser():
	parser = argparse.ArgumentParser(description='Build the splits of remote datasets')
	
	# root setting
	parser.add_argument('--device', type=str, default='cuda:0', help="Device to use for computation (e.g., 'cuda:0', 'cuda:1', or 'cpu')")
	parser.add_argument('--data_dir', default='../datasets', type=str)
	parser.add_argument('--dataset', default='RSSCN7', type=str, choices=['AID', 'UCMerced_LandUse', 'NWPU-RESISC45', 'RSSCN7'])
	parser.add_argument('--class_list', default='dataset/splits-0.2/classInd.txt', type=str)
	parser.add_argument('--train_list', default='dataset/splits-0.2/train_split.txt', type=str)
	parser.add_argument('--val_list', default='dataset/splits-0.2/val_split.txt', type=str)
	parser.add_argument('--splits_ratio', default='0.2', type=str)
	parser.add_argument('--resume_path', default='checkpoints/dataset_ratio_arch_local_mode.pth', type=str)

	# model setting
	parser.add_argument('--arch', default='wide_resnet50_2', type=str, choices=['wide_resnet50_2', 'resnet18', 'shufflenet_v2_x1_5', 'efficientnet_b3'])
	parser.add_argument('--local_arch', default='MobileNetV2', type=str, choices=['MobileNetV2', 'efficientnet_b3'])
	parser.add_argument('--mode', default='kd', type=str, choices=['va', 'kd'])
	parser.add_argument('--batch_size', default=24, type=int)
	parser.add_argument('--img_size', default=224, type=int)
	parser.add_argument('--n_workers', default=8, type=int)
	

	# train setting
	parser.add_argument('--start_epoch', default=0, type=int)
	parser.add_argument('--epochs', default=50, type=int)
	parser.add_argument('--step_size', default=20, type=int)
	parser.add_argument('--lr', default=1e-4, type=float)	

	# kd setting
	parser.add_argument('--temperature', default=3.0, type=float)
	parser.add_argument('--loss_coefficient', default=1.0, type=float)
	args = parser.parse_args()
	return args

