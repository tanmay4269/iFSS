# conda activate iFSS-tmp

# python train.py \
#     --config config/pascal/pascal_split0_resnet50.yaml \
#     --model_name PFENet \
#     > logs/train.log 2>&1 \
#     && ~/.automations/notifier.sh \
#         "pfenet training completed" \
#         || ~/.automations/notifier.sh \
#         "pfenet training failed" \
#         --repeat 3

python train.py \
    --config config/pascal/pascal_split0_simple_fss.yaml \
    --model_name FewShotViTModel \
    > logs/train.log 2>&1 \
    && ~/.automations/TelegramBots/notifier.sh \
        "simple fss training completed" \
        || ~/.automations/TelegramBots/notifier.sh \
        "simple fss training failed" \
        --repeat 3