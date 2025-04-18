import os
import random
import logging
from copy import deepcopy
from collections import defaultdict
from easydict import EasyDict as edict

import cv2
import torch
import numpy as np
from tqdm import tqdm

from albumentations import Normalize
from torch.utils.data import DataLoader

from isegm.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.vis import draw_probmap, draw_points, UndoNormalize
from isegm.utils.misc import save_checkpoint
from isegm.utils.serialization import get_config_repr
from isegm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict
from .optimizer import get_optimizer


class iFSSTrainer(object):
    def __init__(
        self,
        model,
        cfg,
        model_cfg,
        loss_cfg,
        trainset,
        valset,
        optimizer="adam",
        optimizer_params=None,
        image_dump_interval=200,
        checkpoint_interval=10,
        tb_dump_period=25,
        max_interactive_points=0,
        lr_scheduler=None,
        metrics=None,
        additional_val_metrics=None,
        net_inputs=("images", "points"),
        max_num_next_clicks=0,
        click_models=None,
        prev_mask_drop_prob=0.0,
    ):
        self.cfg = cfg
        self.pretraining_enabled = cfg.pretrain_mode

        self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.net_inputs = net_inputs
        self.max_num_next_clicks = max_num_next_clicks

        self.click_models = click_models
        self.prev_mask_drop_prob = prev_mask_drop_prob

        if cfg.distributed:
            cfg.batch_size //= cfg.ngpus
            cfg.val_batch_size //= cfg.ngpus

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ""
        self.sw = None

        self.trainset = trainset
        self.valset = valset

        logger.info(
            f"Dataset of {trainset.get_samples_number()} samples was loaded for training."
        )
        logger.info(
            f"Dataset of {valset.get_samples_number()} samples was loaded for validation."
        )

        shuffle = not (cfg.debug == "one_batch_overfit")
        self.train_data = DataLoader(
            trainset,
            cfg.batch_size,
            sampler=get_sampler(trainset, shuffle=shuffle, distributed=cfg.distributed),
            drop_last=True,
            pin_memory=True,
            num_workers=cfg.workers,
        )

        self.val_data = DataLoader(
            valset,
            cfg.val_batch_size,
            sampler=get_sampler(valset, shuffle=False, distributed=cfg.distributed),
            drop_last=True,
            pin_memory=True,
            num_workers=cfg.workers,
        )

        # For visualizations
        self.undo_normalize = None
        for t in self.trainset.augmentator.transforms:
            if not isinstance(t, Normalize):
                continue
            self.undo_normalize = UndoNormalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225), 
                max_pixel_value=1.0, 
            )

        self.optim = get_optimizer(model, optimizer, optimizer_params)
        model = self._load_weights(model)

        if cfg.multi_gpu:
            model = get_dp_wrapper(cfg.distributed)(
                model, device_ids=cfg.gpu_ids, output_device=cfg.gpu_ids[0]
            )

        # INFO: Too much useless text dumed
        # if self.is_master:
        #     logger.info(model)
        #     logger.info(get_config_repr(model._config))

        self.device = cfg.device
        self.net = model.to(self.device)
        self.lr = optimizer_params["lr"]

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        if self.click_models is not None:
            for click_model in self.click_models:
                for param in click_model.parameters():
                    param.requires_grad = False
                click_model.to(self.device)
                click_model.eval()

    def run(self, num_epochs, start_epoch=None, validation=True):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch

        logger.info(f"Starting Epoch: {start_epoch}")
        logger.info(f"Total Epochs: {num_epochs}")
        for epoch in range(start_epoch, num_epochs):
            self.training(epoch)
            if validation:
                self.validation(epoch)

    def training(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(
                log_dir=str(self.cfg.LOGS_PATH),
                flush_secs=10,
                dump_period=self.tb_dump_period,
            )

        if self.cfg.distributed:
            self.train_data.sampler.set_epoch(epoch)

        log_prefix = "Train" + self.task_prefix.capitalize()
        tbar = (
            tqdm(self.train_data, file=self.tqdm_out, ncols=100)
            if self.is_master else self.train_data
        )

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        self.net.train()
        train_loss = 0.0
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.train_data) + i
            
            loss, losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Logging from here
            losses_logging["overall"] = loss
            reduce_loss_dict(losses_logging)

            train_loss += losses_logging["overall"].item()

            if self.is_master:
                for loss_name, loss_value in losses_logging.items():
                    self.sw.add_scalar(
                        tag=f"{log_prefix}Losses/{loss_name}",
                        value=loss_value.item(),
                        global_step=global_step,
                    )

                for k, v in self.loss_cfg.items():
                    if (
                        "_loss" in k
                        and hasattr(v, "log_states")
                        and self.loss_cfg.get(k + "_weight", 0.0) > 0
                    ):
                        v.log_states(self.sw, f"{log_prefix}Losses/{k}", global_step)

                if (
                    self.image_dump_interval > 0
                    and global_step % self.image_dump_interval == 0
                ):
                    self.save_visualization(
                        splitted_batch_data, outputs, global_step, prefix="train"
                    )

                self.sw.add_scalar(
                    tag=f"{log_prefix}States/learning_rate",
                    value=(
                        self.lr
                        if not hasattr(self, "lr_scheduler")
                        else self.lr_scheduler.get_lr()[-1]
                    ),
                    global_step=global_step,
                )

                tbar.set_description(
                    f"Epoch {epoch}, training loss {train_loss/(i+1):.4f}"
                )
                for metric in self.train_metrics:
                    if metric.name not in ["support_iou", "query_iou"]:
                        continue
                    
                    metric.log_states(
                        self.sw, f"{log_prefix}Metrics/{metric.name}", global_step
                    )
                    
                self.sw.flush()

        if self.is_master:
            for metric in self.train_metrics:
                if metric.name not in ["support_iou", "query_iou"]:
                    continue
                
                self.sw.add_scalar(
                    tag=f"{log_prefix}Metrics/{metric.name}",
                    value=metric.get_epoch_value(),
                    global_step=epoch,
                    disable_avg=True,
                )
                
            self.sw.flush()

        # Saving model ckpt from here
        if self.is_master and self.cfg.debug != 'one_batch_overfit':
            save_checkpoint(
                self.net,
                self.cfg.CHECKPOINTS_PATH,
                prefix=self.task_prefix,
                epoch=None,
                multi_gpu=self.cfg.multi_gpu,
            )

            if isinstance(self.checkpoint_interval, (list, tuple)):
                checkpoint_interval = [
                    x for x in self.checkpoint_interval if x[0] <= epoch
                ][-1][1]
            else:
                checkpoint_interval = self.checkpoint_interval

            if epoch % checkpoint_interval == 0:
                save_checkpoint(
                    self.net,
                    self.cfg.CHECKPOINTS_PATH,
                    prefix=self.task_prefix,
                    epoch=epoch,
                    multi_gpu=self.cfg.multi_gpu,
                )

        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.step()

    def validation(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(
                log_dir=str(self.cfg.LOGS_PATH),
                flush_secs=10,
                dump_period=self.tb_dump_period,
            )

        log_prefix = "Val" + self.task_prefix.capitalize()
        tbar = (
            tqdm(self.val_data, file=self.tqdm_out, ncols=100)
            if self.is_master
            else self.val_data
        )

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.eval()
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.val_data) + i
            
            loss, batch_losses_logging, splitted_batch_data, outputs = (
                self.batch_forward(batch_data, validation=True)
            )

            # Logging
            batch_losses_logging["overall"] = loss
            reduce_loss_dict(batch_losses_logging)
            for loss_name, loss_value in batch_losses_logging.items():
                losses_logging[loss_name].append(loss_value.item())

            val_loss += batch_losses_logging["overall"].item()

            if self.is_master:
                tbar.set_description(
                    f"Epoch {epoch}, validation loss: {val_loss/(i + 1):.4f}"
                )
                for metric in self.val_metrics:
                    if metric.name not in ["support_iou", "query_iou"]:
                        continue
                    
                    metric.log_states(
                        self.sw, f"{log_prefix}Metrics/{metric.name}", global_step
                    )

        # More logging
        if self.is_master:
            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(
                    tag=f"{log_prefix}Losses/{loss_name}",
                    value=np.array(loss_values).mean(),
                    global_step=epoch,
                    disable_avg=True,
                )

            for metric in self.val_metrics:
                if metric.name not in ["support_iou", "query_iou"]:
                    continue
                
                self.sw.add_scalar(
                    tag=f"{log_prefix}Metrics/{metric.name}",
                    value=metric.get_epoch_value(),
                    global_step=epoch,
                    disable_avg=True,
                )

    def batch_forward(self, batch_data, validation=False):
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = dict()

        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            support, query = edict(), edict()

            support.image, support.gt = batch_data["s_images"], batch_data["s_instances"]
            support.points = batch_data["s_points"]
            query.image, query.gt = batch_data["q_images"], batch_data["q_masks"]

            support.prev_output = torch.zeros_like(support.image, dtype=torch.float32)[:, :1, :, :]
            query.prev_output = torch.zeros_like(query.image, dtype=torch.float32)[:, :1, :, :]

            last_click_indx = None
            model_was_training = self.net.training
            self.net.eval()
            with torch.no_grad():  # ! Need to consider this intearction
                if self.cfg.debug == "one_batch_overfit":
                    num_iters = self.max_num_next_clicks
                else:
                    num_iters = random.randint(0, self.max_num_next_clicks)

                for click_indx in range(num_iters):
                    last_click_indx = click_indx

                    if (self.click_models is None or click_indx >= len(self.click_models)):
                        eval_model = self.net
                    else:
                        eval_model = self.click_models[click_indx]

                    outputs = eval_model(support, query, self.pretraining_enabled)

                    # For next iteration
                    support.prev_output = torch.sigmoid(outputs["s_instances"])
                    if not self.pretraining_enabled:
                        query.prev_output = torch.sigmoid(outputs["q_masks"])

                    support.points = get_next_points(
                        support.prev_output, support.gt, support.points, click_indx + 1
                    )

                if (
                    self.net.with_prev_mask
                    and self.prev_mask_drop_prob > 0
                    and last_click_indx is not None
                ):
                    if self.cfg.debug == "one_batch_overfit":
                        zero_mask = torch.zeros_like(support.prev_output[:, :1, :, :]).bool()
                    else:
                        zero_mask = (
                            np.random.random(size=support.prev_output.size(0))
                            < self.prev_mask_drop_prob
                        )
                    support.prev_output[zero_mask] = torch.zeros_like(
                        support.prev_output[zero_mask]
                    )

            if model_was_training and not validation:
                self.net.train()
                
            batch_data["s_points"] = support.points
            outputs = self.net(support, query, self.pretraining_enabled)

            # TODO: Write a new method for this
            loss = 0.0
            loss = self.add_loss(
                "s_instance_loss",
                loss,
                losses_logging,
                validation,
                lambda: (outputs["s_instances"], batch_data["s_instances"]),
            )

            if "s_instances_aux" in outputs:
                loss = self.add_loss(
                    "s_instance_aux_loss",
                    loss,
                    losses_logging,
                    validation,
                    lambda: (outputs["s_instances_aux"], batch_data["s_instances"]),
                )

            if "q_masks" in outputs:
                loss = self.add_loss(
                    "q_mask_loss",
                    loss,
                    losses_logging,
                    validation,
                    lambda: (outputs["q_masks"].float(), batch_data["q_masks"].float()),
                )

            if "q_masks_aux" in outputs:
                loss = self.add_loss(
                    "q_mask_aux_loss",
                    loss,
                    losses_logging,
                    validation,
                    lambda: (outputs["q_masks_aux"], batch_data["q_masks"]),
                )
                
            if "q_masks_aux_list" in outputs:
                loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
                loss_criterion = loss_cfg.get("q_mask_aux_loss")
                loss_weight = loss_cfg.get("q_mask_aux_loss_weight", 0.0)
                q_aux_loss = torch.zeros_like(loss)
                if loss_weight > 0.0:
                    for mask in outputs["q_masks_aux_list"]:
                        mask_loss = loss_criterion(mask, batch_data["q_masks"])
                        mask_loss = torch.mean(loss)
                        q_aux_loss = q_aux_loss + mask_loss
                        
                    q_aux_loss = q_aux_loss / len(outputs["q_masks_aux_list"])
                    q_aux_loss = loss_weight * q_aux_loss
                
                loss = loss + q_aux_loss
                losses_logging["q_mask_aux_loss"] = q_aux_loss


            if self.is_master:
                with torch.no_grad():
                    for m in metrics:
                        m.update(
                            *(outputs.get(x) for x in m.pred_outputs),
                            *(batch_data[x] for x in m.gt_outputs),
                        )

        return loss, losses_logging, batch_data, outputs

    def add_loss(
        self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs
    ):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + "_weight", 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = torch.mean(loss)
            losses_logging[loss_name] = loss
            loss = loss_weight * loss
            total_loss = total_loss + loss

        return total_loss

    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix):
        # TODO: Refactor this method
        if "q_masks" in outputs:
            self._visualize_support_and_query(splitted_batch_data, outputs, global_step, prefix)
        else:
            self._visualize_support_only(splitted_batch_data, outputs, global_step, prefix)    
            
    def _visualize_support_only(self, splitted_batch_data, outputs, global_step, prefix):
        output_images_path = self.cfg.VIS_PATH / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        images = splitted_batch_data['s_images']
        points = splitted_batch_data['s_points']
        instance_masks = splitted_batch_data['s_instances']

        gt_instance_masks = instance_masks.cpu().numpy()
        predicted_instance_masks = torch.sigmoid(outputs['s_instances']).detach().cpu().numpy()
        points = points.detach().cpu().numpy()

        image_blob, points = images[0], points[0]
        gt_mask = np.squeeze(gt_instance_masks[0], axis=0)
        predicted_mask = np.squeeze(predicted_instance_masks[0], axis=0)

        if self.undo_normalize is not None:
            image_blob = self.undo_normalize.apply(image_blob.cpu().numpy())

        image = image_blob * 255
        image = image.transpose((1, 2, 0))

        image_with_points = draw_points(image, points[:self.max_interactive_points], (0, 255, 0))
        image_with_points = draw_points(image_with_points, points[self.max_interactive_points:], (0, 0, 255))

        gt_mask[gt_mask < 0] = 0.25
        gt_mask = draw_probmap(gt_mask)
        predicted_mask = draw_probmap(predicted_mask)
        viz_image = np.hstack((image_with_points, gt_mask, predicted_mask)).astype(np.uint8)

        _save_image('instance_segmentation', viz_image[:, :, ::-1])
        
    def _visualize_support_and_query(self, splitted_batch_data, outputs, global_step, prefix):
        output_images_path = self.cfg.VIS_PATH / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f"{global_step:06d}"

        def _save_image(suffix, image):
            cv2.imwrite(
                str(output_images_path / f"{image_name_prefix}_{suffix}.jpg"),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )

        for k, v in splitted_batch_data.items():
            splitted_batch_data[k] = v.detach().cpu().numpy()

        s_images = splitted_batch_data["s_images"]
        s_points = splitted_batch_data["s_points"]
        s_gt_instance_masks = splitted_batch_data["s_instances"]

        q_images = splitted_batch_data["q_images"]
        q_gt_masks = splitted_batch_data["q_masks"]

        predicted_s_instance_masks = (
            torch.sigmoid(outputs["s_instances"]).detach().cpu().numpy()
        )

        predicted_q_masks = torch.sigmoid(outputs["q_masks"]).detach().cpu().numpy()

        s_image_blob, s_points = s_images[0], s_points[0]
        s_gt_mask = np.squeeze(s_gt_instance_masks[0], axis=0)
        s_predicted_mask = np.squeeze(predicted_s_instance_masks[0], axis=0)

        q_image_blob = q_images[0]
        q_gt_mask = np.squeeze(q_gt_masks[0], axis=0)
        q_predicted_mask = np.squeeze(predicted_q_masks[0], axis=0)

        if self.undo_normalize is not None:
            s_image_blob = self.undo_normalize.apply(s_image_blob)
            q_image_blob = self.undo_normalize.apply(q_image_blob)

        s_image = s_image_blob * 255
        s_image = s_image.transpose((1, 2, 0))

        q_image = q_image_blob * 255
        q_image = q_image.transpose((1, 2, 0))

        s_image_with_points = draw_points(
            s_image, s_points[: self.max_interactive_points], (0, 255, 0)
        ) # +ve clicks
        s_image_with_points = draw_points(
            s_image_with_points, s_points[self.max_interactive_points :], (0, 0, 255)
        ) # -ve clicks

        s_gt_mask[s_gt_mask < 0] = 0.25
        s_gt_mask = draw_probmap(s_gt_mask)
        s_predicted_mask = draw_probmap(s_predicted_mask)
        s_viz_image = np.hstack(
            (s_image_with_points, s_gt_mask, s_predicted_mask)
        ).astype(np.uint8)

        q_gt_mask[q_gt_mask < 0] = 0.25
        q_gt_mask = draw_probmap(q_gt_mask)
        q_predicted_mask = draw_probmap(q_predicted_mask)
        q_viz_image = np.hstack((q_image, q_gt_mask, q_predicted_mask)).astype(np.uint8)

        _save_image("support_instance_segmentation", s_viz_image[:, :, ::-1])
        _save_image("query_segmentation", q_viz_image[:, :, ::-1])

    def _load_weights(self, net):
        if self.cfg.weights is not None:
            if os.path.isfile(self.cfg.weights):
                load_weights(net, self.cfg.weights)
                self.cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{self.cfg.weights}'")
        elif self.cfg.resume_exp is not None:
            checkpoints = list(
                self.cfg.CHECKPOINTS_PATH.glob(f"{self.cfg.resume_prefix}*.pth")
            )
            assert len(checkpoints) == 1

            checkpoint_path = checkpoints[0]
            logger.info(f"Loaded checkpoint from path: {checkpoint_path}")
            load_weights(net, str(checkpoint_path))
        return net

    @property
    def is_master(self):
        return self.cfg.local_rank == 0


def get_next_points(pred, gt, points, click_indx, pred_thresh=0.49):
    assert click_indx > 0
    pred = pred.cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), "constant").astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), "constant").astype(np.uint8)
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))]
            if is_positive:
                points[bindx, num_points - click_indx, 0] = float(coords[0])
                points[bindx, num_points - click_indx, 1] = float(coords[1])
                points[bindx, num_points - click_indx, 2] = float(click_indx)
            else:
                points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                points[bindx, 2 * num_points - click_indx, 2] = float(click_indx)

    return points


def load_weights(model, path_to_weights):
    current_state_dict = model.state_dict()
    new_state_dict = torch.load(path_to_weights, map_location="cpu")["state_dict"]
    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict)
