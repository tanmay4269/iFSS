import argparse
from utils import *

import torch
from torch.utils.data import DataLoader
import torchvision
import albumentations as A

import segmentation_models_pytorch as smp
from models.seg import SegmentationModel

from datasets.sbd import SBDataset


cfg = {
    ### Data ###
    # Dataset
    "data_root": "/workspace/ifss/data/sbd/benchmark_RELEASE/dataset",
    "mode": "segmentation",
    "min_target_frac": 0.1,  # every class with less than 10% area coverage in sampled target will be removed
    "crop_size": (320, 480), # H, W
    
    # Dataloader
    "batch_size": 32,

    ### Training ###
    # Optimizer
    "lr": 3e-4,

    # Trainer
    "epochs": 50,
}

def get_args(cfg):
    parser = argparse.ArgumentParser(description="Training Configuration")

    for key, value in cfg.items():
        arg_type = type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=value)

    args = parser.parse_args()
    return args


class Trainer():
    def __init__(self):
        args = get_args(cfg)
        self.args = args

        crop_size = self.args.crop_size
        train_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0, value=(0,0,0), mask_value=0),
                A.RandomCrop(*crop_size),
            ]
        )

        val_transform = A.Compose(
            [
                A.PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0, value=(0,0,0), mask_value=0),
                A.RandomCrop(*crop_size),
            ]
        )
        
        trainset = SBDataset(args, image_set="train", transforms=train_transform)
        valset = SBDataset(args, image_set="val", transforms=val_transform)

        self.train_loader = DataLoader(
            trainset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=4, 
            # pin_memory=False
        )
        
        self.val_loader = DataLoader(
            valset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=4, 
            # pin_memory=False
        )

        self.model = SegmentationModel(args).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

    def get_accuracy(self, logits, targets):
        predictions = torch.max(logits, dim=1)[1]

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        correct = (predictions == targets).sum().item()
        total = targets.numel()
        accuracy = correct / total

        return accuracy

    def run_epoch(self, data_loader, optimizer=None, validation=False):
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        
        if validation: self.model.eval()
        else: self.model.train()

        with torch.set_grad_enabled(not validation):
            for i, (images, targets) in enumerate(data_loader):
                images = images.cuda()
                targets = targets.cuda()
                
                logits, loss = self.model(images, targets)
                
                if not validation:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_meter.update(loss.item(), images.shape[0])
                
                accuracy = self.get_accuracy(logits, targets)
                accuracy_meter.update(accuracy, images.shape[0])

                if not validation and i % (len(data_loader) // 8) == 0:
                    print(f'\t Iter [{i}/{len(data_loader)}]\t Loss: {loss.item():.4f}\t Acc: {accuracy:.2f}')

        return loss_meter.avg, accuracy_meter.avg

    def run(self):
        epochs = self.args.epochs
        for epoch in range(epochs):
            train_loss, train_acc = self.run_epoch(self.model, self.train_loader, self.optimizer)
            val_loss, val_acc = self.run_epoch(self.model, self.val_loader, validation=True)
            # val_loss, val_acc = run_epoch(model, train_loader, validation=True)

            print(f'Epoch [{epoch}/{epochs}]\t Train Loss: {train_loss:.4f}\t Val Loss: {val_loss:.4f}\t Train Acc: {train_acc:.2f}\t Val Acc: {val_acc:.2f}')


if __name__ == "__main__":
    Trainer().run()