nohup bash -c 'python -u main.py --device cuda:2 --arch shufflenet_v2_x1_5 --mode kd --epochs 50 --temperature 3.0 --loss_coefficient 1.0'> ./shuffle_kd.log 2>&1 &
        
