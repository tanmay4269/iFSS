# iFSS-RITM

This work is built on top of [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation) by extending interactive segmentation to the few shot relm. 

To get things rolling, follow these steps:
1. Set up the environment: `docker pull tanmay4269/ifss:ritm`
    - To manually set it up:
        - `docker pull pytorch/pytorch:1.4-cuda10.1-cudnn7-devel`
        - `apt-get update && apt-get install -y libgl1-mesa-dev && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6`
        - `pip install -r requirements.txt`
2. Download the dataset
    - `wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar`
    - `gdown https://drive.google.com/uc?id=1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF` replace this with `SegmentationClass` in the devkit
    - rename `SegmentationClass` to `SegmentationClassAug`
3. Weights: `https://onedrive.live.com/?authkey=%21AMkPimlmClRvmpw&id=F7FD0B7F26543CEB%21112&cid=F7FD0B7F26543CEB&parId=root&parQt=sharedby&o=OneUp`
    - inside `pretrained_weights`