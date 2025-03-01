export DEBUG=1

python train.py \
    models/ifss_models/pfenet_sbd_ifss.py \
    --gpus=0 \
    --workers=4 \
    --batch-size=8 \
    --exp-name=debug \
    --pretrain-mode \
    --debug=one_batch_overfit