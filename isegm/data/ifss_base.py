import random
import pickle
import numpy as np
import torch
from torchvision import transforms
from .points_sampler import MultiPointSampler
from .sample import DSample


class iFSSDataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        augmentator=None,
        points_sampler=MultiPointSampler(max_num_points=12),
        min_object_area=0,
        keep_background_prob=0.0,
        with_image_info=False,
        epoch_len=-1,
    ):
        super(iFSSDataset, self).__init__()
        self.epoch_len = epoch_len
        self.augmentator = augmentator
        self.min_object_area = min_object_area
        self.keep_background_prob = keep_background_prob
        self.points_sampler = points_sampler
        self.with_image_info = with_image_info
        self.to_tensor = transforms.ToTensor()

        self.dataset_samples = None

    def __getitem__(self, index):
        query_sample, support_sample = self.get_sample(index)

        query_sample = self.augment_sample(query_sample)
        support_sample = self.augment_sample(support_sample)

        # Sample points only on support image
        query_sample.remove_small_objects(self.min_object_area)
        support_sample.remove_small_objects(self.min_object_area)

        # TODO: would preffer instances
        self.points_sampler.sample_object(support_sample)

        points = np.array(self.points_sampler.sample_points())
        mask = self.points_sampler.selected_mask

        output = {
            "s_images": self.to_tensor(support_sample.image),
            "s_instances": mask,
            "s_points": points.astype(np.float32),
            "q_images": self.to_tensor(query_sample.image),
            "q_masks": query_sample._encoded_masks.transpose(2, 0, 1),
        }

        return output

    def augment_sample(self, sample) -> DSample:
        if self.augmentator is None:
            return sample

        valid_augmentation = False
        while not valid_augmentation:
            sample.augment(self.augmentator)
            keep_sample = (
                self.keep_background_prob < 0.0
                or random.random() < self.keep_background_prob
            )
            valid_augmentation = len(sample) > 0 or keep_sample

        return sample

    def get_sample(self, index) -> DSample:
        raise NotImplementedError

    def __len__(self):
        if self.epoch_len > 0:
            return self.epoch_len
        else:
            return self.get_samples_number()

    def get_samples_number(self):
        return len(self.dataset_samples)
