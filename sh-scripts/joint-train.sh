export DEBUG=1
python train.py \
    models/ifss_models/hrnet18_sbd_ifss.py \
    --gpus=0 \
    --workers=4 \
    --batch-size=2 \
    --exp-name=ifss-training