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

        self.train_loader = DataLoader(
            self.train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )

        self.val_loader = DataLoader(
            self.val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
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

            print(f"Epoch [{epoch+1}/{epochs}]", end="\t")
            for k, v in train_logs.items():
                print(f"Train {k.capitalize()}: {v:.4f}", end="\t")

            print("\n\t\t", end="")
            for k, v in val_logs.items():
                print(f"Val {k.capitalize()}: {v:.4f}", end="\t")

            print()

            # saving
            if epoch > 0 and np.mean(val_losses) > val_logs["loss"]:
                torch.save(self.model.state_dict(), self.weights_save_path)
                print(f"Model saved at epoch {epoch+1}")

            val_losses.append(val_logs["loss"])

    def run_epoch(
        self, data_loader, optimizer=None, validation=False, early_stopping=False
    ):
        loss_meter = AverageMeter()
        iou_meter = AverageMeter()
        noc85_meter = AverageMeter()
        noc90_meter = AverageMeter()

        if validation:
            self.model.eval()
        else:
            self.model.train()

        with torch.set_grad_enabled(not validation):
            for i, (x_q, y_q, x_s, y_s) in enumerate(data_loader):
                x_q, y_q = x_q.cuda(), y_q.cuda()
                x_s, y_s = x_s[:, 0].cuda(), y_s[:, 0].cuda()

                s_prev_pred = torch.zeros_like(y_s)
                q_prev_pred = torch.zeros_like(y_q)

                s_click_mask = torch.zeros_like(y_s)  # BHW
                s_click_mask = s_click_mask.unsqueeze(1).repeat(1, 2, 1, 1)  # B2HW

                # For NoC calculation
                # reached_85 = torch.full((x_s.shape[0],), float("inf"))
                # reached_90 = reached_85.clone()

                for click_idx in range(self.args.max_click_iters):
                    output = self.model(
                        s_click_mask, s_prev_pred, x_s, y_s, q_prev_pred, x_q, y_q
                    )

                    s_logits, q_logits = output['logits']
                    _, _, loss = output['losses']

                    s_prev_pred = s_preds = torch.max(s_logits, dim=1)[1]
                    q_prev_pred = torch.max(q_logits, dim=1)[1]

                    self.update_click_masks(
                        s_click_mask, s_preds, y_s, is_first_click=(click_idx == 0)
                    )

                    if not validation:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # logging
                    # loss_meter.update(loss.item(), images.shape[0])

                    # ious = get_iou(preds, targets)

                    # above_85_idx = ((ious > 0.85).int() == 1).cpu()
                    # above_90_idx = ((ious > 0.90).int() == 1).cpu()
                    # reached_85[above_85_idx] = torch.min(
                    #     reached_85[above_85_idx], torch.tensor(click_idx)
                    # )
                    # reached_90[above_90_idx] = torch.min(
                    #     reached_90[above_90_idx], torch.tensor(click_idx)
                    # )

                    # iou = ious.mean().item()
                    # if click_idx == 0 or (click_idx + 1) % 5 == 0:
                    #     print(
                    #         f"\t\tClick idx [{click_idx+1}/{self.args.max_click_iters}]\t IoU: {iou:.2f}"
                    #     )

                    # early break
                    # if early_stopping and torch.min(ious) > 0.95:
                    #     break

                # iou = get_iou(preds, targets).mean().item()
                # iou_meter.update(iou, images.shape[0])

                # noc85_meter.update(mean_ignore_inf(reached_85))
                # noc90_meter.update(mean_ignore_inf(reached_90))

                # if (
                #     not validation
                #     and len(data_loader) // 8 > 0
                #     and i % (len(data_loader) // 8) == 0
                # ):
                #     print(
                #         f"\t Iter [{i}/{len(data_loader)}]\t Loss: {loss.item():.4f}\t IoU: {iou:.2f}"
                #     )

        return {
            "loss": loss_meter.avg,
            "iou": iou_meter.avg,
            "noc85": noc85_meter.avg,
            "noc90": noc90_meter.avg,
        }

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

    def get_dataset(self, dataset_name):
        if dataset_name == "sbd_val":
            dataset = self.valset
        elif dataset_name == "davis":
            dataset = DavisDataset(self.args, self.val_transform)
        else:
            raise f"Dataset {dataset_name} isn't implemented"

        return dataset

    def visualize(self, dataset_name, max_click_iters):
        loader = DataLoader(
            self.get_dataset(dataset_name),
            batch_size=1,
            shuffle=True,
        )

        self.model.eval()
        with torch.no_grad():
            for images, targets in loader:
                # model needs batch size > 1
                images = torch.cat((images, images), dim=0)
                targets = torch.cat((targets, targets), dim=0)
                images, targets = images.cuda(), targets.cuda()

                prev_pred = torch.zeros_like(targets)
                click_masks = torch.zeros_like(targets)  # BHW
                click_masks = click_masks.unsqueeze(1).repeat(1, 2, 1, 1)  # B2HW

                for click_idx in range(max_click_iters):
                    self.update_click_masks(
                        click_masks, prev_pred, targets, is_first_click=(click_idx == 0)
                    )

                    logits, _ = self.model(images, click_masks, prev_pred, targets)

                    prev_pred = preds = torch.max(logits, dim=1)[1]

                    # Plot images with click masks
                    iou = (
                        get_iou(preds[0].unsqueeze(0), targets[0].unsqueeze(0))
                        .mean()
                        .item()
                    )  # ignore the 2nd image
                    self.plot_images_with_clicks(
                        images, targets, preds, click_masks, click_idx, iou
                    )

                break

    def plot_images_with_clicks(
        self, images, targets, preds, click_masks, click_idx, iou
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        i = 0  # ignore the 2nd image
        # Displaying ground truth
        axes[0].imshow(images[i].cpu().numpy().transpose(1, 2, 0))
        axes[0].imshow(targets[i].cpu().numpy(), cmap="gray", alpha=0.5)
        axes[0].axis("off")

        # Displaying prediction and clicks
        _click_masks = click_masks[i].cpu()
        overlay = np.zeros((_click_masks.shape[1], _click_masks.shape[2], 4))

        overlay[..., 0] = _click_masks[0]  # Negative clicks = Red
        overlay[..., 1] = _click_masks[1]  # Positive clicks = Green
        overlay[..., 3] = np.maximum(_click_masks[0], _click_masks[1])  # Alpha channel

        axes[1].imshow(images[i].cpu().numpy().transpose(1, 2, 0))
        axes[1].imshow(preds[i].cpu().numpy(), cmap="gray", alpha=0.5)
        axes[1].imshow(overlay)
        axes[1].axis("off")

        fig.suptitle(f"Iteration {click_idx+1} | IoU: {iou:.2f}")
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
