export DEBUG=1

# python train.py \
#     models/ifss_models/hrnet18_sbd_ifss.py \
#     --gpus=0 \
#     --workers=4 \
#     --batch-size=2 \
#     --exp-name=ifss-training

python train.py \
    models/ifss_models/hrnet18_sbd_ifss.py \
    --gpus=0 \
    --workers=8 \
    --batch-size=22 \
    --exp-name=pretraining \
    --pretrain-mode

# python train.py \
#     models/ifss_models/hrnet18_sbd_ifss.py \
#     --gpus=0 \
#     --workers=4 \
#     --batch-size=20 \
#     --weights=experiments/ifss_models/sbd_hrnet18/000_pretraining/checkpoints/last_checkpoint.pth \
#     --exp-name=ifss-training