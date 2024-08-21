import numpy as np
from pathlib import Path

from albumentations import cv2

from utils import *
from .custom_dataset import CustomDataset


class DavisDataset(CustomDataset):
    def __init__(self, args, transforms):
        super().__init__(args, transforms)

        self.data_root = Path(DAVIS_DATAROOT)

        self._images_path = self.data_root / 'img'
        self._insts_path = self.data_root / 'gt'

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.*')}

        self.images = []

        for image_name in self.dataset_samples:
            mask_path = str(self._masks_paths[image_name.split('.')[0]])

            labels, counts = np.unique(cv2.imread(mask_path)[:,:,-1], return_counts=True)
            
            filtered_labels = []
            fracs = counts / (self.crop_size[0] * self.crop_size[1])
            for label, frac in zip(labels, fracs):
                if label > 0 and frac > self.min_target_frac:
                    filtered_labels.append(label)

            if len(filtered_labels) > 0:
                self.images.append(image_name)


    
    def get_sample(self, index):
        image_name = self.images[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split('.')[0]])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)[:,:,-1]

        # cv2.imwrite('./img.png', image)
        # cv2.imwrite('./mask.png', mask)

        # cv2.imwrite('./mask_0.png', mask[:,:,0])
        # cv2.imwrite('./mask_1.png', mask[:,:,1])
        # cv2.imwrite('./mask_2.png', mask[:,:,2])

        return image, mask
