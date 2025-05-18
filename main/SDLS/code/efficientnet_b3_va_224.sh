nohup bash -c 'python -u main.py --device cuda:0 --arch efficientnet_b3 --local_arch MobileNetV2 --mode va --img_size 224 --epochs 50' > ./efficientnet_b3_va_224.log 2>&1 &
