import os
import numpy as np
from PIL import Image
from scipy.io import loadmat

from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from albumentations import cv2

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
        self.args = args
        self.image_set = image_set
        self.transforms = AlbumentationsTransformWrapper(transforms)
        self.mode = args.mode
        self.num_classes = 20

        self.min_target_frac = args.min_target_frac
        self.crop_size = args.crop_size

        self._get_target = (
            self._get_segmentation_target
            if self.mode == "segmentation"
            else self._get_boundaries_target
        )

        self.images, self.masks = self.prepare_dataset()

    def prepare_dataset(self):
        images, masks = [], []

        sbd_root = self.args.data_root
        image_dir = os.path.join(sbd_root, "img")
        mask_dir = os.path.join(sbd_root, "cls")

        split_f = os.path.join(sbd_root, self.image_set.rstrip("\n") + ".txt")

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

    def __getitem__(self, index):
        image_path = self.images[index]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = self._get_target(self.masks[index])

        img, target = self.transforms(img, target)

        labels, counts = np.unique(target, return_counts=True)

        filtered_labels = []        
        fracs = counts / (self.crop_size[0] * self.crop_size[1])
        for label, frac in zip(labels, fracs):
            if label > 0 and frac > self.min_target_frac:
                filtered_labels.append(label)

        # print(len(filtered_labels), end=', ')
        chosen_label = np.random.choice(filtered_labels) 
        chosen_target = (target == chosen_label).astype(float)

        image, target = self.transforms.to_tensor(img, chosen_target)
    
        return image.float() / 255.0, target.long()

    def __len__(self):
        return len(self.images)