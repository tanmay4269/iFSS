import numpy as np
from torch.utils.data import Dataset

from utils import *


class CustomDataset(Dataset):
    def __init__(self, args, transforms):
        self.args = args

        self.transforms = AlbumentationsTransformWrapper(transforms)

        self.min_target_frac = args.min_target_frac
        self.crop_size = args.crop_size


    def __getitem__(self, index):
        img, target = self.get_sample(index)

        img, target = self.transforms(img, target)

        labels, counts = np.unique(target, return_counts=True)

        filtered_labels = []
        fracs = counts / (self.crop_size[0] * self.crop_size[1])
        for label, frac in zip(labels, fracs):
            if label > 0 and frac > 0: #self.min_target_frac:
                filtered_labels.append(label)

        chosen_label = np.random.choice(filtered_labels) 
        target = (target == chosen_label).astype(float)

        image, target = self.transforms.to_tensor(img, target)
    
        return image.float() / 255.0, target.long()


    def __len__(self):
        return len(self.images)