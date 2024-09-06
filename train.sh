# Pre-training
# python train.py \
#     models/ifss_models/hrnet18_sbd_ifss.py \
#     --gpus=0 \
#     --workers=4 \
#     --batch-size=8 \
#     --weights=experiments/ifss_models/sbd_hrnet18/000_fss-pretraining/checkpoints/last_checkpoint.pth \
#     --exp-name=fss-pretraining \
#     --fss-pretrain


# iFSS training -- post-pretraining
python train.py \
    models/ifss_models/hrnet18_sbd_ifss.py \
    --gpus=0 \
    --workers=4 \
    --batch-size=20 \
    --weights=experiments/ifss_models/sbd_hrnet18/000_fss-pretraining/checkpoints/last_checkpoint.pth \
    --exp-name=ifss-training