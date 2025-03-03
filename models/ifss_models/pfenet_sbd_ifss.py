import torch
from functools import partial
from easydict import EasyDict as edict
from albumentations import *

from isegm.engine.ifss_trainer import iFSSTrainer

from isegm.model import initializer
from isegm.model.losses import *
from isegm.model.metrics import AdaptiveIoU
from isegm.model.ifss_pfenet_model import PFENetModel

from isegm.data.transforms import *
from isegm.data.points_sampler import MultiPointSampler
from isegm.data.datasets.fss_sbd import iFSS_SBD_Dataset

MODEL_NAME = "sbd_pfenet_ifss"


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    # model_cfg.crop_size = (320, 480)
    model_cfg.crop_size = (473, 473)
    
    model_cfg.num_max_points = 24

    model = PFENetModel(
        with_aux_output=False,
        use_rgb_conv=False,
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_lr_mult=1.0,
    )

    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type="gaussian", magnitude=2.0))

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size
    
    if cfg.debug == "one_batch_overfit":
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    loss_cfg = edict()
    # loss_cfg.s_instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.s_instance_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.s_instance_loss_weight = 1.0
    # loss_cfg.s_instance_aux_loss = SigmoidBinaryCrossEntropyLoss()
    # loss_cfg.s_instance_aux_loss_weight = 0.4

    if not cfg.pretrain_mode:
        # loss_cfg.q_mask_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
        loss_cfg.q_mask_loss = SigmoidBinaryCrossEntropyLoss()
        loss_cfg.q_mask_loss_weight = 1.0
        # loss_cfg.q_mask_aux_loss = SigmoidBinaryCrossEntropyLoss()
        # loss_cfg.q_mask_aux_loss_weight = 0.4

    train_augmentator = Compose(
        [
            UniformRandomResize(scale_range=(0.75, 1.25)),
            Flip(),
            RandomRotate90(),
            ShiftScaleRotate(border_mode=0, p=0.75),
            PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
            RandomCrop(*crop_size),
            # CenterCrop(*crop_size),
            RandomBrightnessContrast(),
            RGBShift(
                r_shift_limit=(-10, 10),
                g_shift_limit=(-10, 10),
                b_shift_limit=(-10, 10),
            ),
            # HueSaturationValue(p=0.25),
            # GaussianBlur(p=0.25),
            Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225), 
                max_pixel_value=1.0
            ),
        ], 
        p=1.0
    )

    val_augmentator = Compose(
        [
            UniformRandomResize(scale_range=(0.75, 1.25)),
            PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
            RandomCrop(*crop_size),
            Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225), 
                max_pixel_value=1.0),
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
        )

    optimizer_params = {"lr": 3e-4, "betas": (0.9, 0.999), "eps": 1e-8}

    lr_scheduler = partial(
        torch.optim.lr_scheduler.MultiStepLR, milestones=[50], gamma=0.5
    )
    
    eval_metrics = [
        AdaptiveIoU(name="support_iou", pred_output="s_instances", gt_output="s_instances")
    ]
    if not cfg.pretrain_mode:
        eval_metrics.append(
            AdaptiveIoU(name="query_iou", pred_output="q_instances", gt_output="q_instances")
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
        image_dump_interval=50,  # FIXME: units?
        metrics=eval_metrics,
        max_interactive_points=model_cfg.num_max_points,
        max_num_next_clicks=3,
    )
    trainer.run(num_epochs=220)
