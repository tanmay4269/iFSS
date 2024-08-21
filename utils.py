import sys
import argparse

import torch

SBD_DATAROOT = "/workspace/data/sbd/benchmark_RELEASE/dataset"

DAVIS_DATAROOT = "/workspace/data/DAVIS"

def get_args(cfg):
    parser = argparse.ArgumentParser()
    
    # Adding arguments from the cfg dictionary
    for key, value in cfg.items():
        arg_type = type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=value)
    
    try:
        # Parse arguments, ignoring unrecognized arguments
        args, unknown = parser.parse_known_args()
    except SystemExit:
        print("Error parsing arguments.")
        sys.exit(1)
    
    return args

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

def mean_ignore_inf(tensor):
    # Replace inf with NaN
    tensor = torch.where(torch.isinf(tensor), torch.nan, tensor)
    
    # Compute the mean ignoring NaNs
    return torch.nanmean(tensor)

def get_iou(preds, targets):
    preds = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    intersection = (preds & targets).sum(dim=1).float()
    union = (preds | targets).sum(dim=1).float()

    iou = intersection / torch.clamp(union, min=1.0)
    iou[union == 0] = float('nan')

    return iou