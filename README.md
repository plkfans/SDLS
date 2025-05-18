# SDLS
This is the Pytorch implementation of SDLS for remote sensing scene image classification.


# 📁 Project Structure
<pre>./SDLS/
├── datasets
└── code</pre>

`datasets/`:  Contains remote sensing image datasets used for training and evaluation. For example, RSSCN7, AID, etc.

`code/`: Contains the implementation of the SDLS architecture, including training scripts, model definitions, configuration files, and evaluation tools.

# ⚙️ Environment
All the experiments are implemented by Pytorch 1.12.1 Version in the computing environment of 1 × 24-GB NVIDIA GeForce RTX 3090 GPU. In order to obtain reliable experimental results, we randomly divided the dataset using the same training ratio on each dataset, repeated the experiment five times, and reported the mean and standard deviation of these results. Model weights are saved based on the best overall accuracy of the SDLS.

Create a conda environment with the name SDLS and Python 3.8 in the environment：
```bash
conda create -n SDLS python=3.8 
conda activate SDLS
```
Install PyTorch and torchvision：
```bash
pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cu113
```
Install additional dependencies:
```bash
pip install opencv-python
pip install pycm
pip install matplotlib
pip install scikit-learn
```
# 🚀 Usage example: Training CNN_MobileNetV2 on RSSCN7

## 🔹 Data preparation
The dataset should be organized in the following directory structure:

<pre>./datasets/RSSCN7/
│
├── images/                  
│   ├── Grass
│   ├── Field
│   ├── Industry
│   └── ...
└── splits</pre>

Split the dataset by a specific percentage (e.g. 20% training, 80% testing):
```bash
cd ./code
python build_list.py --data_dir ../datasets/RSSCN7/images --out_dir ../datasets/RSSCN7/splits-0.2 --train_ratio 0.2
```

## 🔹 Training model
The training script supports two modes:

`va (Vanilla)`: Standard training of the CNN backbone without SDLS. This serves as the baseline.

`kd (Knowledge Distillation)`: Trains the CNN backbone with the proposed SDLS (Self-Distillation and Local Stream) framework enabled.

Example: Training ResNet18 in `va` Mode：
```bash
chmod +x R18_va.sh
./R18_va.sh
```
Training ResNet18 in `kd` Mode：
```bash
chmod +x R18_kd.sh
./R18_kd.sh
```
## 🔹 Using Our Pretrained Weights
If you would like to use our pretrained weights, please follow these steps:

1. Download the checkpoint file corresponding to the dataset. The `train.log` and `model.pth` of each result are provided on [Baidu Cloud](https://pan.baidu.com/s/1khux_Y-eO1oxEut36T4gMw?pwd=6666).

2. Place the `model.pth` in the `./checkpoints/` folder.

3. Run the test script to generate the confusion matrix:
```bash
python test.py --device cuda:0 --arch resnet18 --mode va
python test.py --device cuda:0 --arch resnet18 --mode kd
```
4. You can also perform fine-tuning or continue training from the pretrained weights:
```bash
./R18_kd.sh
```
The model will continue training from the model weights saved from the epoch with the highest OA (Overall Accuracy) on the validation set.
# 📖 Citation

# 🔗 Acknowledgements

The dataset splitting strategy and confusion matrix plotting are adapted from:
[SKAL](https://github.com/hw2hwei/SKAL).

We sincerely thank the authors of these repositories for releasing their code as open source, which greatly supported and inspired our research work.

