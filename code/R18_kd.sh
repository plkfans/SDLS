nohup bash -c 'python -u main.py --device cuda:2 --arch resnet18 --mode kd --epochs 50 --temperature 3.0 --loss_coefficient 1.0'> ./R18_kd.log 2>&1 &
        
