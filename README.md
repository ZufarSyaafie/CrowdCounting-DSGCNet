# A Dual-Stream Graph Convolutional Network for Crowd Counting via Feature Correlation Mining
## Overview🔍
**Abstract -** Deep learning-based crowd counting methods have achieved remarkable progress in recent years. However, in complex crowd scenarios, existing models still face challenges when adapting to significant density distribution differences between regions. Additionally, the inconsistency of individual representations caused by viewpoint changes and body posture differences further limits the counting accuracy of the models. To address these challenges, we propose DSGC-Net, a Dual-Stream Graph Convolutional Network based on feature correlation mining. DSGC-Net introduces a Density Approximation (DA) branch and a Representation Approximation (RA) branch. By constructing and modeling semantic graphs, it captures the inconsistencies in density variations and representation distributions. The DA branch incorporates a density prediction module that generates density distribution maps, and constructs a density-driven semantic graph based on density similarity. The RA branch establishes a representation-driven semantic graph by computing global representation similarity. Then, graph convolutional networks are used to model the features of the two semantic graphs separately, mining latent semantic relationships, which enhance the model’s ability to adapt to density variations and improve counting accuracy in multi-view and multi-pose scenarios. Extensive experiments on three benchmarks demonstrate that DSGC-Net outperforms current state-of-the-art methods. In particular, we achieve MAE of 48.9 and 5.9 in ShanghaiTech PartA and PartB datasets, respectively.
## Datasets📚
We evaluate the proposed method on three of the most widely used crowd counting datasets: ShanghaiTech PartA and PartB and UCF-QNRF.
## Experimental Results🏆

## Getting Started🚀
### 1. Data Preparation
First, prepare the dataset. Then, run the following command to generate the corresponding ground truth density distribution maps:
```
python density_data_preparation/k_nearest_gaussian_kernel.py
```
Immediately afterward, We use a list file to collect all the input images and their ground truth annotations in a counting dataset. When your dataset is organized as recommended in the following, the format of this list file is defined as:
```
train/scene01/img01.jpg train/scene01/img01.txt
train/scene01/img02.jpg train/scene01/img02.txt
...
train/scene02/img01.jpg train/scene02/img01.txt
```
#### Dataset structures:
```
DATA_ROOT/
        |->train/
        |    |->scene01/
        |    |->scene02/
        |    |->...
        |->test/
        |    |->scene01/
        |    |->scene02/
        |    |->...
        |->train.list
        |->test.list
        |->gt_density_maps/
        |    |->train/
        |    |->test/  
```
### 2. Training
The network can be trained using the train.py script. For training on ShanghaiTech PartA, use
```
CUDA_VISIBLE_DEVICES=0 python train.py --data_root $DATA_ROOT \
    --dataset_file SHHA \
    --epochs 3500 \
    --lr_drop 3500 \
    --output_dir ./logs \
    --checkpoints_dir ./weights \
    --tensorboard_dir ./logs \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 8 \
    --eval_freq 1 \
    --gpu_id 0
```
## Cite our work📝
```Coming soon...```
## License📜
The source code is free for research and education use only. Any comercial use should get formal permission first.
