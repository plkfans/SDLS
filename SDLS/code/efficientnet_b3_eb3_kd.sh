nohup bash -c 'python -u main.py --device cuda:2 --arch efficientnet_b3 --local_arch efficientnet_b3 --mode kd --img_size 300 --epochs 50 --temperature 3.0 --loss_coefficient 1.0'> ./efficientnet_b3_kd_300.log 2>&1 &
        
