export DEBUG=1

# python train.py \
#     models/ifss_models/pfenet_sbd_ifss.py \
#     --gpus=0 \
#     --workers=1 \
#     --batch-size=4 \
#     --exp-name=debug \
#     --pretrain-mode \
#     --debug=one_batch_overfit

python train.py \
    models/ifss_models/hrnet18_sbd_ifss.py \
    --gpus=0 \
    --workers=1 \
    --batch-size=4 \
    --exp-name=debug \
    --pretrain-mode \
    --debug=one_batch_overfit

# python train.py \
#     models/ifss_models/hrnet18_sbd_ifss.py \
#     --gpus=0 \
#     --workers=8 \
#     --batch-size=22 \
#     --exp-name=expt-01 \
#     --pretrain-mode
