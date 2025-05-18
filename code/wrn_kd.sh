nohup bash -c 'python -u main.py --device cuda:2 --arch wide_resnet50_2 --mode kd --epochs 50 --temperature 3.0 --loss_coefficient 1.0'> ./wrn_kd.log 2>&1 &
        
