nohup bash -c 'python -u main.py --device cuda:0 --arch efficientnet_b3 --local_arch efficientnet_b3 --mode va --batch_size 16 --img_size 330 --epochs 50' > ./efficientnet_b3_va_330.log 2>&1 &
