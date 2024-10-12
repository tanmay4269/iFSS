# iFSS-RITM

This work build on top of [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation) by extending interactive segmentation to few shot setting. 

To get things rolling, follow these steps:
1. Set up the environment
    - `docker pull pytorch/pytorch:1.4-cuda10.1-cudnn7-devel`
    - `apt-get update && apt-get install -y libgl1-mesa-dev && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6`
    - `pip install -r requirements.txt`
2. Download the dataset
    -  `wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar`
    -  rename `SegmentationClass` to `SegmentationClassAug`
    -  copy lists: `sbd_data.txt` and `val.txt` from here: `https://github.com/dvlab-research/PFENet/tree/master/lists/pascal`