nohup bash -c 'python -u main.py --device cuda:0 --arch efficientnet_b3 --local_arch efficientnet_b3 --mode va --img_size 300 --epochs 50' > ./efficientnet_b3_va_300.log 2>&1 &
