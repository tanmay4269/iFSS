import torch
from functools import partial
from easydict import EasyDict as edict
from albumentations import *

from isegm.engine.ifss_trainer import iFSSTrainer

from isegm.model import initializer
from isegm.model.losses import *
from isegm.model.metrics import AdaptiveIoU
from isegm.model.ifss_hrnet_model import iFSS_HRNetModel

from isegm.data.transforms import *
from isegm.data.points_sampler import MultiPointSampler
from isegm.data.datasets.fss_sbd import iFSS_SBD_Dataset

MODEL_NAME = "sbd_hrnet18"


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (320, 480)
    model_cfg.num_max_points = 24

    model = iFSS_HRNetModel(
        width=18,
        ocr_width=64,
        with_aux_output=True,
        use_leaky_relu=True,
        use_rgb_conv=False,
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        norm_layer=nn.BatchNorm2d
    )

    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type="gaussian", magnitude=2.0))

    for net in [model.support_net, model.query_net]:
        net.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.HRNETV2_W18)

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    # loss_cfg.s_instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.s_instance_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.s_instance_loss_weight = 1.0
    loss_cfg.s_instance_aux_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.s_instance_aux_loss_weight = 0.4

    # loss_cfg.q_mask_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.q_mask_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.q_mask_loss_weight = 1.0
    loss_cfg.q_mask_aux_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.q_mask_aux_loss_weight = 0.4

    if cfg.debug == 'one_batch_overfit':
        train_augmentator = Compose(
            [
                UniformRandomResize(scale_range=(0.75, 1.25)),
                Flip(),
                RandomRotate90(),
                ShiftScaleRotate(
                    shift_limit=0.03,
                    scale_limit=0,
                    rotate_limit=(-3, 3),
                    border_mode=0,
                    p=0.75,
                ),
                PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
                RandomCrop(*crop_size),
                # RandomBrightnessContrast(
                #     brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.25
                # ),
                # RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.25),
            ],
            p=1.0,
        )
    else:
        train_augmentator = Compose(
            [
                UniformRandomResize(scale_range=(0.75, 1.25)),
                Flip(),
                RandomRotate90(),
                ShiftScaleRotate(
                    shift_limit=0.03,
                    scale_limit=0,
                    rotate_limit=(-3, 3),
                    border_mode=0,
                    p=0.75,
                ),
                PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
                RandomCrop(*crop_size),
                RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.25
                ),
                RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.25),
            ],
            p=1.0,
        )

    val_augmentator = Compose(
        [
            UniformRandomResize(scale_range=(0.75, 1.25)),
            PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
            RandomCrop(*crop_size),
        ],
        p=1.0,
    )

    points_sampler = MultiPointSampler(
        model_cfg.num_max_points,
        prob_gamma=0.80,
        merge_objects_prob=0.15,
        max_num_merged_objects=2,
    )

    trainset = iFSS_SBD_Dataset(
        cfg,  # FIXME: Find a better way of doing this
        data_root=cfg.SBD_TRAIN_PATH,
        data_list=cfg.SBD_TRAIN_LIST,
        mode="train",
        split=0,
        use_coco=False,
        use_split_coco=False,
        augmentator=train_augmentator,
        min_object_area=80,
        keep_background_prob=0.01,
        points_sampler=points_sampler,
        samples_scores_gamma=1.25,
    )

    if cfg.debug == "one_batch_overfit":
        valset = trainset
    else:
        valset = iFSS_SBD_Dataset(
            cfg,
            data_root=cfg.SBD_TRAIN_PATH,
            data_list=cfg.SBD_VAL_LIST,
            mode="val",
            split=0,
            use_coco=False,
            use_split_coco=False,
            augmentator=val_augmentator,
            min_object_area=80,
            keep_background_prob=0.01,
            points_sampler=points_sampler,
            samples_scores_gamma=1.25,
        )

    optimizer_params = {"lr":5e-4, "betas": (0.9, 0.999), "eps": 1e-8}

    lr_scheduler = partial(
        torch.optim.lr_scheduler.MultiStepLR, milestones=[50], gamma=0.1
    )
    trainer = iFSSTrainer(
        model,
        cfg,
        model_cfg,
        loss_cfg,
        trainset,
        valset,
        optimizer="adam",
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        checkpoint_interval=[(0, 20), (100, 10)],  # (epoch_num, interval)
        image_dump_interval=100,  # FIXME: units?
        metrics=[
            AdaptiveIoU(name="support_iou", pred_output="s_instances", gt_output="s_instances"),
            AdaptiveIoU(name="query_iou", pred_output="q_masks", gt_output="q_masks"),
        ],
        max_interactive_points=model_cfg.num_max_points,
        max_num_next_clicks=3,
    )
    trainer.run(num_epochs=220)
