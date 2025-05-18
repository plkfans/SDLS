nohup bash -c 'python -u main.py --device cuda:0 --arch wide_resnet50_2 --mode va --epochs 50' > ./wrn_va.log 2>&1 &
