export DEBUG=1

python scripts/evaluate_ifss_model.py \
    NoBRS \
    --checkpoint /home/tvg/Projects/iFSS_saved_expts/2024-12-29-07-47-42_pretraining/checkpoints/last_checkpoint.pth \
    --datasets SBD-iFSS \
    --gpus 0 \
    --vis-preds \
    --save-ious \
    --print-ious 