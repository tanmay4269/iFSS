# iFSS-RITM

This work is built on top of [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation) by extending interactive segmentation to the few shot relm. 

To get things rolling, follow these steps:
1. Set up the environment: `docker pull tanmay4269/ifss:ritm`
    - To manually set it up:
        - `docker pull pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel`
        - `apt-get update && apt-get install -y libgl1-mesa-dev && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6`
        - `pip install -r requirements.txt`
2. Download the dataset
    - `wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar`
    - `gdown https://drive.google.com/uc?id=1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF` replace this with `SegmentationClass` in the devkit
    - rename `SegmentationClass` to `SegmentationClassAug`
    ```
    sudo apt update
    sudo apt -y install wget unzip libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6
    pip install gdown
    
    mkdir data
    cd data
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    tar -xvf VOCtrainval_11-May-2012.tar
    rm -fr VOCtrainval_11-May-2012.tar

    gdown https://drive.google.com/uc?id=1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF
    unzip SegmentationClass.zip
    rm -fr SegmentationClass.zip

    mv SegmentationClass VOCdevkit/VOC2012/SegmentationClassAug
    cd ..

    git clone https://tanmay4269:<token>@github.com:tanmay4269/iFSS-RITM.git

    cd iFSS-RITM
    rm -rf data/*devkit
    ln -s ~/data/VOCdevkit/ ~/iFSS-RITM/data/VOCdevkit

    mkdir pretrained_models
    cd pretrained_models
    wget https://opr0mq.dm.files.1drv.com/y4mIoWpP2n-LUohHHANpC0jrOixm1FZgO2OsUtP2DwIozH5RsoYVyv_De5wDgR6XuQmirMV3C0AljLeB-zQXevfLlnQpcNeJlT9Q8LwNYDwh3TsECkMTWXCUn3vDGJWpCxQcQWKONr5VQWO1hLEKPeJbbSZ6tgbWwJHgHF7592HY7ilmGe39o5BhHz7P9QqMYLBts6V7QGoaKrr0PL3wvvR4w
    mv * hrnetv2_w18_imagenet_pretrained.pth
    cd ..

    conda init
    source ~/.bashrc
    conda create -n ifss python=3.7 -y
    conda activate ifss
    pip install -r requirements.txt
    sh sh-scripts/pretrain+ifss.sh 
    ```
3. Weights: `https://onedrive.live.com/?authkey=%21AMkPimlmClRvmpw&id=F7FD0B7F26543CEB%21112&cid=F7FD0B7F26543CEB&parId=root&parQt=sharedby&o=OneUp`
    - inside `pretrained_weights`
