import os
import numpy as np
from scipy.io import loadmat

from albumentations import cv2

from utils import *
from .custom_dataset import CustomDataset, AlbumentationsTransformWrapper

class SBDataset(CustomDataset):
    def __init__(self, args, image_set, transforms):
        super().__init__(args, transforms)
        
        self.image_set = image_set
        self.mode = args.mode
        self.num_classes = 20

        self._get_target = (
            self._get_segmentation_target
            if self.mode == "segmentation"
            else self._get_boundaries_target
        )

        self.data_root = SBD_DATAROOT

        self.images, self.masks = self.prepare_dataset()

    def prepare_dataset(self):
        images, masks = [], []

        image_dir = os.path.join(self.data_root, "img")
        mask_dir = os.path.join(self.data_root, "cls")

        split_f = os.path.join(self.data_root, self.image_set.rstrip("\n") + ".txt")

        with open(os.path.join(split_f)) as fh:
            file_names = [x.strip() for x in fh.readlines()]

        for idx in range(0, len(file_names), int(1/self.args.dataset_frac)):
            image_path = os.path.join(image_dir, file_names[idx] + ".jpg")
            mask_path = os.path.join(mask_dir, file_names[idx] + ".mat")
            
            target = self._get_target(mask_path)
            _, target = self.transforms(target, target)

            labels, counts = np.unique(target, return_counts=True)

            filtered_labels = []            
            fracs = counts / (self.crop_size[0] * self.crop_size[1])
            for label, frac in zip(labels, fracs):
                if label > 0 and frac > self.min_target_frac:
                    filtered_labels.append(label)

            if len(filtered_labels) > 0:
                images.append(image_path)
                masks.append(mask_path)

        return images, masks

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
    
    def get_sample(self, index):
        image_path = self.images[index]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = self._get_target(self.masks[index])

        return img, target
