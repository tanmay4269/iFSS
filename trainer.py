import os
import collections

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import albumentations as A

from utils import *
from models.models import iSegModel, iFSSModel

from datasets.sbd import SBDataset
from datasets.davis import DavisDataset
from datasets.fss_dataset import FSSDataset

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class Trainer:
    def __init__(self, cfg):
        args = get_args(cfg)
        self.args = args

        crop_size = self.args.crop_size
        self.train_transform = A.Compose(
            [
                A.Resize(*crop_size),
            ]
        )

        self.val_transform = A.Compose(
            [
                A.Resize(*crop_size),
            ]
        )

        self.train_data = FSSDataset(
            split=args.split,
            shot=args.shot,
            data_root=args.data_root,
            data_list=args.train_list,
            transform=self.train_transform,
            mode="train",
            use_coco=args.use_coco,
            use_split_coco=args.use_split_coco,
        )

        self.val_data = FSSDataset(
            split=args.split,
            shot=args.shot,
            data_root=args.data_root,
            data_list=args.val_list,
            transform=self.val_transform,
            mode="val",
            use_coco=args.use_coco,
            use_split_coco=args.use_split_coco,
        )

        print(f"Train Data Size: {len(self.train_data)}\t Val Data Size: {len(self.val_data)}")

        self.train_loader = DataLoader(
            self.train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=False,
        )

        self.val_loader = DataLoader(
            self.val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
        )

        # self.model = iSegModel(args).cuda()
        self.model = iFSSModel(args).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        # Logging and Checkpointing
        self.weights_save_path = args.weights_save_path
        if args.ckpt_path:
            self.model.load_state_dict(torch.load(args.ckpt_path, weights_only=True))

    def run(self):
        val_losses = collections.deque(maxlen=5)
        epochs = self.args.epochs

        print("Starting Training...")
        for epoch in range(epochs):
            train_logs = self.run_epoch(self.train_loader, self.optimizer)
            val_logs = self.run_epoch(self.val_loader, validation=True)

            # Logging
            print(f"Epoch [{epoch+1}/{epochs}]", end="\t")
            for k, v in train_logs.items():
                print(f"Train {k.capitalize()}: {v:.4f}", end="\t")

            print("\n\t\t", end="")
            for k, v in val_logs.items():
                print(f"Val {k.capitalize()}: {v:.4f}", end="\t")

            print()

            # Checkpointing
            if epoch > 0 and np.mean(val_losses) > val_logs["loss"]:
                torch.save(self.model.state_dict(), self.weights_save_path)
                print(f"Model saved at epoch {epoch+1}")

            val_losses.append(val_logs["s_loss"] + val_logs["q_loss"])

    def run_epoch(
        self,
        data_loader,
        optimizer=None,
        validation=False,
        early_stopping=False,
        visualize=False,
        max_click_iters=None,
    ):
        if max_click_iters is None:
            max_click_iters = self.args.max_click_iters

        if visualize:
            validation = True
        else:
            log_dict = self.logger("init")

        if validation:
            self.model.eval()
        else:
            self.model.train()

        with torch.set_grad_enabled(not validation):
            for i, (x_q, y_q, x_s, y_s) in enumerate(data_loader):
                x_s, y_s = x_s[:, 0].cuda(), y_s[:, 0].cuda()
                x_q, y_q = x_q.cuda(), y_q.cuda()

                s_prev_pred = torch.zeros_like(y_s)
                q_prev_pred = torch.zeros_like(y_q)

                s_click_mask = (
                    torch.zeros_like(y_s).unsqueeze(1).repeat(1, 2, 1, 1)
                )  # BHW -> B2HW

                # For NoC calculation
                s_reached_85 = torch.full((x_s.shape[0],), float("inf"))
                s_reached_90 = s_reached_85.clone()

                q_reached_85 = torch.full((x_q.shape[0],), float("inf"))
                q_reached_90 = q_reached_85.clone()

                for click_idx in range(max_click_iters):
                    output = self.model(
                        s_click_mask, s_prev_pred, x_s, y_s, q_prev_pred, x_q, y_q
                    )

                    s_logits, q_logits = output["logits"]
                    s_loss, q_loss, loss = output["losses"]

                    s_prev_pred = s_preds = torch.max(s_logits, dim=1)[1]
                    q_prev_pred = q_preds = torch.max(q_logits, dim=1)[1]

                    self.update_click_masks(
                        s_click_mask, s_preds, y_s, is_first_click=(click_idx == 0)
                    )

                    if not validation:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # Logging
                    s_ious = get_iou(s_preds, y_s)
                    q_ious = get_iou(q_preds, y_q)

                    for ious, reached_85, reached_90 in zip(
                        [s_ious, q_ious],
                        [s_reached_85, s_reached_90],
                        [q_reached_85, q_reached_90],
                    ):
                        above_85_idx = ((ious > 0.85).int() == 1).cpu()
                        above_90_idx = ((ious > 0.90).int() == 1).cpu()
                        reached_85[above_85_idx] = torch.min(
                            reached_85[above_85_idx], torch.tensor(click_idx)
                        )
                        reached_90[above_90_idx] = torch.min(
                            reached_90[above_90_idx], torch.tensor(click_idx)
                        )

                    s_iou = s_ious.mean().item()
                    q_iou = q_ious.mean().item()
                    if click_idx == 0 or (click_idx + 1) % 5 == 0:
                        print(
                            f"\t\tClick idx [{click_idx+1}/{self.args.max_click_iters}]\t S-IoU: {s_iou:.2f}\t Q-IoU: {q_iou:.2f}"
                        )

                    # Visualizing
                    if visualize:
                        self._visualize(
                            s_click_mask,
                            s_preds,
                            x_s,
                            y_s,
                            q_preds,
                            x_q,
                            y_q,
                            click_idx,
                            s_iou,
                            q_iou,
                        )
                    else:
                        log_dict["s_loss"].update(s_loss.item(), x_s.shape[0])
                        log_dict["q_loss"].update(q_loss.item(), x_q.shape[0])
                        log_dict["loss"].update(loss.item(), x_s.shape[0])

                    # Early break
                    # if early_stopping and min(torch.min(s_ious), torch.min(q_ious)) > 0.95:
                    #     break

                if visualize:
                    return

                log_dict["s_iou"].update(s_iou, x_s.shape[0])
                log_dict["q_iou"].update(q_iou, x_q.shape[0])

                log_dict["s_noc85"].update(mean_ignore_inf(s_reached_85))
                log_dict["s_noc90"].update(mean_ignore_inf(s_reached_90))

                log_dict["q_noc85"].update(mean_ignore_inf(q_reached_85))
                log_dict["q_noc90"].update(mean_ignore_inf(q_reached_90))

                if (
                    not validation
                    and len(data_loader) // 8 > 0
                    and i % (len(data_loader) // 8) == 0
                ):
                    print(
                        f"\t Iter [{i}/{len(data_loader)}]\t Loss: {loss.item():.4f}\t S-IoU: {s_iou:.2f}\t Q-IoU: {q_iou:.2f}"
                    )

                # Debug
                break

        return self.logger("return", log_dict)

    def evaluate_dataset(self, dataset_name):
        dataset = self.get_dataset(dataset_name)

        data_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
        )

        logs = self.run_epoch(data_loader, validation=True)

        for k, v in logs.items():
            if k == "loss":
                continue
            print(f"Train {k.capitalize()}: {v:.4f}", end="\t")

    def visualize(self, dataset_name, max_click_iters):
        loader = DataLoader(
            self.get_dataset(dataset_name),
            batch_size=2,
            shuffle=True,
        )

        self.run_epoch(loader, visualize=True, max_click_iters=max_click_iters)

    def get_dataset(self, dataset_name):
        if dataset_name == "sbd_train":
            dataset = self.train_data
        elif dataset_name == "sbd_val":
            dataset = self.val_data
        elif dataset_name == "davis":
            dataset = DavisDataset(self.args, self.val_transform)
        else:
            raise f"Dataset {dataset_name} isn't implemented"

        return dataset

    def _visualize(
        self, s_click_mask, s_pred, x_s, y_s, q_pred, x_q, y_q, click_idx, s_iou, q_iou
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10, 4))

        i = 0  # ignore the 2nd image

        for j, image, mask, pred, iou in zip(
            [0, 1], [x_s, x_q], [y_s, y_q], [s_pred, q_pred], [s_iou, q_iou]
        ):
            # Displaying ground truth
            axes[0, j].imshow(image[i].cpu().numpy().transpose(1, 2, 0))
            axes[0, j].imshow(mask[i].cpu().numpy(), cmap="gray", alpha=0.5)
            axes[0, j].axis("off")
            axes[0, j].set_title("Support Image" if j == 0 else "Query Image")

            # Displaying prediction and clicks
            _click_mask = s_click_mask[i].cpu()

            overlay = np.zeros((_click_mask.shape[1], _click_mask.shape[2], 4))
            if j == 0:
                overlay[..., 0] = _click_mask[0]  # Negative clicks = Red
                overlay[..., 1] = _click_mask[1]  # Positive clicks = Green
                overlay[..., 3] = np.maximum(
                    _click_mask[0], _click_mask[1]
                )  # Alpha channel

            axes[1, j].imshow(image[i].cpu().numpy().transpose(1, 2, 0))
            axes[1, j].imshow(pred[i].cpu().numpy(), cmap="gray", alpha=0.5)
            axes[1, j].imshow(overlay)
            axes[1, j].axis("off")
            axes[1, j].set_title(f"IoU: {iou:.2f}")

        fig.suptitle(f"Iteration {click_idx+1}")
        plt.show()

    def update_click_masks(
        self, prev_click_masks, preds, targets, radius=3, is_first_click=False
    ):
        # batch of updated click_masks
        clicks = self.sample_clicks(preds, targets, is_first_click)

        indices = []
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i**2 + j**2 <= radius**2:
                    indices.append((i, j))

        indices = torch.tensor(indices)
        batches = clicks.shape[0]
        for i in range(batches):
            _sum = sum(clicks[i])

            _indices = indices + torch.sign(_sum) * clicks[i].repeat(
                indices.shape[0], 1
            )
            _indices = _indices.int()

            c = (_sum > 0).int()

            ys = torch.clamp(_indices[:, 0], min=0, max=self.args.crop_size[0] - 1)
            xs = torch.clamp(_indices[:, 1], min=0, max=self.args.crop_size[1] - 1)

            prev_click_masks[i, c, ys, xs] = 1

    def sample_clicks(self, preds, targets, is_first_click=False):
        # batch of single click per pred
        and_masks = torch.logical_and(preds, targets)
        fn_masks = torch.logical_and(targets, torch.logical_not(and_masks))
        fp_masks = torch.logical_and(preds, torch.logical_not(and_masks))

        batches = targets.shape[0]
        num_fn = torch.sum(fn_masks.view(batches, -1), dim=1)
        num_fp = torch.sum(fp_masks.view(batches, -1), dim=1)

        if is_first_click:
            is_pos_click = torch.ones((batches,))
        else:
            is_pos_click = (num_fn > num_fp).float()

        clicks = torch.zeros((batches, 2))

        for i in range(batches):
            if is_first_click or torch.sum(and_masks[i]).item() == 0:
                masks = targets[i]
            else:
                masks = fn_masks[i] if is_pos_click[i] else fp_masks[i]

            indices = torch.nonzero(masks == 1)

            if indices.shape[0] == 0:
                indices = torch.nonzero(targets[i] == 1)

            click = indices[torch.randint(0, indices.shape[0], (1,))]
            clicks[i] = click * torch.sign(is_pos_click[i] - 0.5)

        return clicks

    def logger(self, mode, log_dict=None):
        if mode == "init":
            log_dict = {}

            for k in [
                "s_loss",
                "q_loss",
                "loss",
                "s_iou",
                "q_iou",
                "s_noc85",
                "q_noc85",
                "s_noc90",
                "q_noc90",
            ]:
                log_dict[k] = AverageMeter()

            return log_dict

        if mode == "return":
            ret = {}
            for k, v in log_dict.items():
                ret[k] = v.avg

            return ret


if __name__ == "__main__":
    cfg = {
        ### Data ###
        # Dataset
        "split": 0,
        "shot": 1,
        "data_root": "data/VOCdevkit/VOC2012",
        # "data_root": "/workspace/custom-approaches/interactive-seg/data/VOCdevkit/VOC2012",
        "train_list": "datasets/lists/pascal/sbd_data.txt",  # voc_sbd_merge_noduplicate.txt
        "val_list": "datasets/lists/pascal/val.txt",
        "use_coco": False,
        "use_split_coco": False,
        # Augmentations
        "crop_size": (320, 480),  # HW
        # Dataloader
        "batch_size": 32,
        "num_workers": 4,
        ### Training ###
        "lr": 3e-4,
        # Trainer
        "epochs": 10,
        "max_click_iters": 20,  # number of times new clicks are sampled
        ### Logging and Checkpointing ###
        "weights_save_path": "./weights/model_1.pth",
        "ckpt_path": None,
        # "ckpt_path": "./weights/model_1.pth",
    }

    Trainer(cfg).run()

    # Trainer(cfg).evaluate_dataset(dataset_name='davis')
