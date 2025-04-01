# iFSS: Interactive Few-Shot Segmentation

## Overview

iFSS (Interactive Few-Shot Segmentation) is an advanced computer vision framework that combines interactive segmentation with few-shot learning approaches for image segmentation tasks. The project extends the RITM (Reviving Iterative Training with Mask Guidance) framework to enable effective segmentation with minimal user input and few examples.

## Key Features

- **Few-Shot Learning**: Ability to learn from just a handful of labeled examples
- **Interactive Segmentation**: Progressive refinement through user interaction points
- **Support-Query Architecture**: Utilizes support images to segment query images
- **Multiple Backbone Options**: Includes HRNet and PFENet-based architectures
- **Extensible Framework**: Easy to integrate new backbone models and segmentation strategies

## Model Architecture

The iFSS framework consists of several key components:

- **Support Branch**: Processes support images and extracts features for few-shot learning
- **Query Branch**: Applies learned features to segment new query images
- **Interactive Module**: Processes user clicks to refine segmentation results
- **Trainers**: Custom training loops for various scenarios (pretraining, fine-tuning)

## Requirements

- Python 3.7+
- PyTorch 1.13.1+
- CUDA 11.6+ (for GPU acceleration)
- Additional dependencies in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/iFSS.git
cd iFSS

# Set up the environment (using Docker - recommended)
docker pull tanmay4269/ifss:ritm

# OR manually set up:
docker pull pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
apt-get update && apt-get install -y libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6
pip install -r requirements.txt
```

## Dataset Setup

```bash
# Download and prepare PASCAL VOC dataset
mkdir data
cd data

# Download VOC dataset
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar

# Download augmented segmentation masks
gdown https://drive.google.com/uc?id=1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF
unzip SegmentationClass.zip
mv SegmentationClass VOCdevkit/VOC2012/SegmentationClassAug

cd ..
```

## Pretrained Models

Pre-trained weights are available for download:
[Pretrained Models](https://onedrive.live.com/?authkey=%21AMkPimlmClRvmpw&id=F7FD0B7F26543CEB%21112&cid=F7FD0B7F26543CEB&parId=root&parQt=sharedby&o=OneUp)

Place the downloaded weights in the `pretrained_models` directory.

## Training

```bash
# Example command to train the HRNet-based iFSS model
python train.py models/ifss_models/hrnet18_sbd_ifss.py

# Example command for PFENet-based model with pretraining
python train.py models/ifss_models/pfenet_sbd_ifss.py --pretrain-mode
```

## Inference

Interactive segmentation can be performed using the provided scripts:

```bash
# Sample inference command
python scripts/evaluate_model.py --model hrnet18_sbd_ifss --checkpoint /path/to/checkpoint
```

## Results

The iFSS framework achieves significant improvements over baseline methods:
- Better segmentation accuracy with fewer user interactions
- Effective few-shot learning for novel classes
- Robust performance across different datasets

## Acknowledgments

This work builds upon [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation) for interactive segmentation and extends it to the few-shot learning paradigm.
