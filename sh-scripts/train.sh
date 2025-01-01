export DEBUG=1

# python train.py \
#     models/ifss_models/hrnet18_sbd_ifss.py \
#     --gpus=0 \
#     --workers=4 \
#     --batch-size=2 \
#     --exp-name=ifss-training

# python train.py \
#     models/ifss_models/hrnet18_sbd_ifss.py \
#     --gpus=0 \
#     --workers=8 \
#     --batch-size=22 \
#     --exp-name=experiment_2 \
#     --pretrain-mode \

python train.py \
    models/ifss_models/hrnet18_sbd_ifss.py \
    --gpus=0 \
    --workers=8 \
    --batch-size=22 \
    --exp-name=experiment_2 \
    --pretrain-mode \
    --resume-exp=016_experiment_2 \
    --resume-prefix=060 \
    --start-epoch=60

# python train.py \
#     models/ifss_models/hrnet18_sbd_ifss.py \
#     --gpus=0 \
#     --workers=1 \
#     --batch-size=22 \
#     --exp-name=one_batch_overfit \
#     --pretrain-mode \
#     --debug="one_batch_overfit"

# python train.py \
#     models/ifss_models/hrnet18_sbd_ifss.py \
#     --gpus=0 \
#     --workers=4 \
#     --batch-size=20 \
#     --weights=experiments/ifss_models/sbd_hrnet18/000_pretraining/checkpoints/last_checkpoint.pth \
#     --exp-name=ifss-training