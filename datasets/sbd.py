import os
import numpy as np
from PIL import Image
from scipy.io import loadmat

from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class AlbumentationsTransformWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, target):
        transformed = self.transform(image=np.array(image), mask=np.array(target))
        return transformed["image"], transformed["mask"]

    def to_tensor(self, image, target):
        transformed = ToTensorV2()(image=np.array(image), mask=np.array(target))
        return transformed["image"], transformed["mask"]

class SBDataset(Dataset):
    def __init__(self, args, image_set, transforms):
        self.image_set = image_set
        self.transforms = AlbumentationsTransformWrapper(transforms)
        self.mode = args.mode
        self.num_classes = 20

        self.min_target_frac = args.min_target_frac
        self.crop_size = args.crop_size

        sbd_root = args.data_root
        image_dir = os.path.join(sbd_root, "img")
        mask_dir = os.path.join(sbd_root, "cls")

        split_f = os.path.join(sbd_root, image_set.rstrip("\n") + ".txt")

        with open(os.path.join(split_f)) as fh:
            file_names = [x.strip() for x in fh.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".mat") for x in file_names]

        self._get_target = (
            self._get_segmentation_target
            if self.mode == "segmentation"
            else self._get_boundaries_target
        )

    def _get_segmentation_target(self, filepath):
        mat = loadmat(filepath)
        return np.array(mat["GTcls"][0]["Segmentation"][0])

    def _get_boundaries_target(self, filepath):
        mat = loadmat(filepath)
        return np.concatenate(
            [
                np.expand_dims(mat["GTcls"][0]["Boundaries"][0][i][0].toarray(), axis=0)
                for i in range(self.num_classes)
            ],
            axis=0,
        )

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        target = self._get_target(self.masks[index])

        img, target = self.transforms(img, target)

        labels, counts = np.unique(target, return_counts=True)

        filtered_labels = []
        
        max_frac_cls = (-1, -1)  # cls, frac
        fracs = counts / (self.crop_size[0] * self.crop_size[1])
        for label, frac in zip(labels, fracs):
            if label > 0 and frac > self.min_target_frac:
                if max_frac_cls[1] < frac: 
                    max_frac_cls = (label, frac)
                
                filtered_labels.append(label)

        if len(filtered_labels) == 0:
            chosen_label = max_frac_cls[0]
        else:
            chosen_label = np.random.choice(filtered_labels) 

        chosen_target = (target == chosen_label).astype(float)

        image, target = self.transforms.to_tensor(img, chosen_target)
    
        return image.float(), target.long()

    def __len__(self):
        return len(self.images)