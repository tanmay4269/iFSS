import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import albumentations as A

from utils import *
from datasets.sbd import SBDataset
from models.seg import SegmentationModel


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Trainer():
    def __init__(self, cfg):
        args = get_args(cfg)
        self.args = args

        crop_size = self.args.crop_size
        train_transform = A.Compose(
            [
                A.Resize(height=crop_size[0], width=crop_size[1], always_apply=True, interpolation=A.cv2.INTER_LINEAR),  
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.RandomRotate90(p=0.5),
                A.PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0, value=(0,0,0), mask_value=0),
                # A.RandomCrop(*crop_size),
            ]
        )

        val_transform = A.Compose(
            [
                A.Resize(height=crop_size[0], width=crop_size[1], always_apply=True, interpolation=A.cv2.INTER_LINEAR),  
                A.PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0, value=(0,0,0), mask_value=0),
                # A.RandomCrop(*crop_size),
            ]
        )
        
        self.trainset = SBDataset(args, image_set="train", transforms=train_transform)
        self.valset = SBDataset(args, image_set="val", transforms=val_transform)

        self.train_loader = DataLoader(
            self.trainset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=self.args.num_workers, 
        )
        
        self.val_loader = DataLoader(
            self.valset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=self.args.num_workers, 
        )

        self.model = SegmentationModel(args).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)


    def sample_clicks(self, preds, targets):
        # batch of single click per pred
        and_masks = torch.logical_and(preds, targets)
        fn_masks = torch.logical_and(targets, torch.logical_not(and_masks))
        fp_masks = torch.logical_and(preds, torch.logical_not(and_masks))

        batches = targets.shape[0]
        num_fn = torch.sum(fn_masks.view(batches, -1), dim=1)
        num_fp = torch.sum(fp_masks.view(batches, -1), dim=1)
        
        is_pos_click = (num_fn > num_fp).float()
        clicks = torch.zeros((batches, 2))

        for i in range(batches):
            if torch.sum(and_masks[i]).item() == 0:
                masks = targets[i]
            else:
                masks = fn_masks[i] if is_pos_click[i] else fp_masks[i]
            
            indices = torch.nonzero(masks == 1)

            if indices.shape[0] == 0:
                indices = torch.nonzero(targets[i] == 1)

            click = indices[torch.randint(0, indices.shape[0], (1,))]
            clicks[i] = click * torch.sign(is_pos_click[i] - 0.5)

        return clicks
    
    def update_click_masks(self, prev_click_masks, preds, targets, radius=3):
        # batch of updated click_masks
        clicks = self.sample_clicks(preds, targets)

        indices = []
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i**2 + j**2 <= radius**2:
                    indices.append((i, j))

        indices = torch.tensor(indices)
        batches = clicks.shape[0]
        for i in range(batches):
            _sum = sum(clicks[i])

            _indices = indices + torch.sign(_sum) * clicks[i].repeat(indices.shape[0], 1)
            _indices = _indices.int()

            c = (_sum > 0).int()

            ys = torch.clamp(_indices[:, 0], min=0, max=self.args.crop_size[0]-1)
            xs = torch.clamp(_indices[:, 1], min=0, max=self.args.crop_size[1]-1)
            
            prev_click_masks[i, c, ys, xs] = 1


    def get_iou(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        correct = (preds == targets).sum().item()
        total = targets.numel()
        iou = correct / total

        return iou

    def run_epoch(self, data_loader, optimizer=None, validation=False):
        loss_meter = AverageMeter()
        iou_meter = AverageMeter()
        
        if validation: self.model.eval()
        else: self.model.train()

        with torch.set_grad_enabled(not validation):
            for i, (images, targets) in enumerate(data_loader):
                images = images.cuda()
                targets = targets.cuda()

                prev_pred = torch.zeros_like(targets)
                click_masks = torch.zeros_like(targets) # BHW
                click_masks = click_masks.unsqueeze(1).repeat(1,2,1,1) # B2HW

                for click_idx in range(self.args.max_click_iters):
                    logits, loss = self.model(images, click_masks, prev_pred, targets)

                    preds = torch.max(logits, dim=1)[1]
                    prev_pred = preds
                    self.update_click_masks(click_masks, preds, targets)
                    
                    if not validation:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # logging
                    loss_meter.update(loss.item(), images.shape[0])
                    
                    if click_idx % 5 == 0: 
                        print(f"\t\tClick idx [{click_idx}/{self.args.max_click_iters}]\t IoU: {self.get_iou(preds, targets):.2f}")

                    
                iou = self.get_iou(preds, targets)
                iou_meter.update(iou, images.shape[0])

                if not validation and i % (len(data_loader) // 8) == 0:
                    print(f'\t Iter [{i}/{len(data_loader)}]\t Loss: {loss.item():.4f}\t IoU: {iou:.2f}')

        return loss_meter.avg, iou_meter.avg

    def run(self):
        epochs = self.args.epochs

        print("Starting Training...")
        for epoch in range(epochs):
            train_loss, train_acc = self.run_epoch(self.train_loader, self.optimizer)
            val_loss, val_acc = self.run_epoch(self.val_loader, validation=True)

            print(f'Epoch [{epoch}/{epochs}]\t Train Loss: {train_loss:.4f}\t Val Loss: {val_loss:.4f}\t Train Acc: {train_acc:.2f}\t Val Acc: {val_acc:.2f}')

    def plot_images_with_clicks(self, images, targets, preds, click_masks, click_idx):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for i in range(2):
            axes[i, 0].imshow(images[i].cpu().numpy().transpose(1, 2, 0))
            axes[i, 0].imshow(targets[i].cpu().numpy(), cmap='gray', alpha=0.5)
            axes[i, 0].set_title(f'Image {i+1}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(images[i].cpu().numpy().transpose(1, 2, 0))
            axes[i, 1].imshow(preds[i].cpu().numpy(), cmap='gray', alpha=0.5)
            axes[i, 1].imshow(click_masks[i, 0].cpu().numpy(), cmap='jet', alpha=0.5)
            axes[i, 1].set_title(f'Mask {i+1} with Click Masks (Iteration {click_idx})')
            axes[i, 1].axis('off')

        plt.show()

    def visualize(self, max_click_iters):
        loader = DataLoader(
            self.valset,
            batch_size=2,
            shuffle=True,
        )

        self.model.eval()
        with torch.no_grad():
            for images, targets in loader:
                images, targets = images.cuda(), targets.cuda()
                prev_pred = torch.zeros_like(targets)
                click_masks = torch.zeros_like(targets) # BHW
                click_masks = click_masks.unsqueeze(1).repeat(1,2,1,1) # B2HW

                for click_idx in range(max_click_iters):
                    logits, _ = self.model(images, click_masks, prev_pred, targets)

                    preds = torch.max(logits, dim=1)[1]
                    prev_pred = preds
                    self.update_click_masks(click_masks, preds, targets)
                    
                    # Plot images with click masks
                    self.plot_images_with_clicks(images, targets, preds, click_masks, click_idx)

                    # logging
                    print(f"\t\tClick idx [{click_idx}/{max_click_iters}]\t IoU: {self.get_iou(preds, targets):.2f}")

                break

                    


if __name__ == "__main__":
    cfg = {
        ### Data ###
        # Dataset
        "data_root": "/workspace/ifss/data/sbd/benchmark_RELEASE/dataset",
        "mode": "segmentation",
        "dataset_frac": 1.0,
        "min_target_frac": 0.05,  # every class with less than "x" 
                                # area coverage in sampled target 
                                # will be removed                              
        "crop_size": (320, 480), # HW
        
        # Dataloader
        "batch_size": 32,
        "num_workers": 4,

        ### Training ###
        "lr": 3e-4,

        # Trainer
        "epochs": 1,
        "max_click_iters": 20,  # number of times new clicks are sampled
    }

    Trainer().run(cfg)