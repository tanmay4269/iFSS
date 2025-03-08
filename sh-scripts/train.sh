export DEBUG=1

# python train.py \
#     models/ifss_models/pfenet_sbd_ifss.py \
#     --gpus=0 \
#     --workers=1 \
#     --batch-size=4 \
#     --exp-name=debug \
#     --debug=one_batch_overfit \
#     --pretrain-mode

python train.py \
    models/ifss_models/pfenet_sbd_ifss.py \
    --gpus=0 \
    --workers=1 \
    --batch-size=4 \
    --exp-name=debug-b4 \
    --debug=one_batch_overfit \
    --weights="weights/pfenet_ifss_pretraining.pth" \
    # --pretrain-mode \

# python train.py \
#     models/ifss_models/pfenet_sbd_ifss.py \
#     --gpus=0 \
#     --workers=8 \
#     --batch-size=80 \
#     --weights="weights/pfenet_ifss_pretraining.pth" \
#     --exp-name=ifss-expt-00 \
#     # --exp-name=debug \