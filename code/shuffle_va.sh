nohup bash -c 'python -u main.py --device cuda:0 --arch shufflenet_v2_x1_5 --mode va --epochs 50' > ./shuffle_va.log 2>&1 &
