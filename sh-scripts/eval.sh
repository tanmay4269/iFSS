export DEBUG=0
python scripts/evaluate_ifss_model.py \
    NoBRS \
    --checkpoint /workspace/iFSS-expts/000_fss-pretraining/checkpoints/last_checkpoint.pth \
    --datasets SBD-iFSS \
    --gpus 0 \
    --vis-preds \
    --save-ious \
    --print-ious 