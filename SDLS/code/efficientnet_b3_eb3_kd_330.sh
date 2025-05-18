nohup bash -c 'python -u main.py --device cuda:2 --arch efficientnet_b3 --local_arch efficientnet_b3 --mode kd --batch_size 16 --img_size 330 --epochs 50 --temperature 3.0 --loss_coefficient 1.0'> ./efficientnet_b3_kd_330.log 2>&1 &
        
