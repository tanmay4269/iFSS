import os
from tqdm import tqdm

import random
import pickle as pkl
from pathlib import Path
from collections import defaultdict

import cv2
from PIL import Image
import numpy as np
from scipy.io import loadmat

import torch 
from torch.utils.data import Dataset

from isegm.utils.misc import get_bbox_from_mask, get_labels_with_sizes
from isegm.data.ifss_base import iFSSDataset
from isegm.data.sample import DSample


class iFSS_SBD_Dataset(iFSSDataset):
    def __init__(self, 
                 data_root, 
                 data_list,
                 mode='train', 
                 split=0, 
                 supports=1,
                 queries=1,
                 use_coco=False,
                 use_split_coco=False,
                 buggy_mask_thresh=0.08, 
                 **kwargs):
        super(iFSS_SBD_Dataset, self).__init__(**kwargs)

        self.mode = mode
        self.supports = supports
        self.queries = queries

        self.data_root = Path(data_root)
        self._buggy_objects = dict()
        self._buggy_mask_thresh = buggy_mask_thresh

        # TODO: write this better, this implementation 
        # is probably not standard
        self.make_dataset(
            mode,
            use_coco,
            use_split_coco,
            split,
            data_root,
            data_list,
        )
        
    def get_sample(self, index):
        image_path, label_path = self.dataset_samples[index]
        image, label = self.get_raw_image_label(image_path, label_path) 

        label_class = np.unique(label).tolist()
        label_class = list(set(label_class) - {0, 255})
        
        filtered_label_class = []
        for c in label_class:
            if c in self.sub_val_list and self.mode in ['val', 'test']:
                filtered_label_class.append(c)
            if c in self.sub_list and self.mode == 'train':
                filtered_label_class.append(c)
        label_class = filtered_label_class    
        assert len(label_class) > 0

        # making a new query label
        class_chosen = random.choice(label_class)
        label = self.choose_label(label, class_chosen)
        
        # selecting support pair
        file_class_chosen = self.sub_class_file_list[class_chosen]
        num_files = len(file_class_chosen)

        while True:
            support_idx = random.randint(0, num_files - 1)
            support_image_path, support_label_path = file_class_chosen[support_idx]
            
            if (
                support_image_path != image_path 
                or support_label_path != label_path
            ):
                break

        support_image, support_label = self.get_raw_image_label(support_image_path, support_label_path)
        support_label = self.choose_label(support_label, class_chosen)

        # query_instances_mask = self.remove_buggy_masks(index, label)
        query_instances_mask = label
        query_instances_ids, _ = get_labels_with_sizes(query_instances_mask)
        
        # support_instances_mask = self.remove_buggy_masks(support_idx, support_label)
        support_instances_mask = support_label
        support_instances_ids, _ = get_labels_with_sizes(support_instances_mask)
        
        return (
            # Query
            DSample(
                image, query_instances_mask, 
                objects_ids=[query_instances_ids[0]], sample_id=index
            ),

            # Support
            DSample(
                support_image, support_instances_mask, 
                objects_ids=[support_instances_ids[0]], sample_id=support_idx
            ),
        )
    
    def remove_buggy_masks(self, index, instances_mask):
        if self._buggy_mask_thresh > 0.0:
            buggy_image_objects = self._buggy_objects.get(index, None)
            if buggy_image_objects is None:
                buggy_image_objects = []
                instances_ids, _ = get_labels_with_sizes(instances_mask)
                for obj_id in instances_ids:
                    obj_mask = instances_mask == obj_id
                    mask_area = obj_mask.sum()
                    bbox = get_bbox_from_mask(obj_mask)
                    bbox_area = (bbox[1] - bbox[0] + 1) * (bbox[3] - bbox[2] + 1)
                    obj_area_ratio = mask_area / bbox_area
                    if obj_area_ratio < self._buggy_mask_thresh:
                        buggy_image_objects.append(obj_id)

                self._buggy_objects[index] = buggy_image_objects
            for obj_id in buggy_image_objects:
                instances_mask[instances_mask == obj_id] = 0

        return instances_mask
    
    def make_dataset(
            self,
            mode='train',
            use_coco=False,
            use_split_coco=False,
            split=0,
            data_root=None,
            data_list=None,
            ):
        assert split in [0, 1, 2, 3, 10, 11, 999]

        if not use_coco:  # Pascal and/or SBD
            class_list = list(range(1, 21))
            if split == 3: 
                sub_list = list(range(1, 16))
                sub_val_list = list(range(16, 21))
            elif split == 2:
                sub_list = list(range(1, 11)) + list(range(16, 21))
                sub_val_list = list(range(11, 16))
            elif split == 1:
                sub_list = list(range(1, 6)) + list(range(11, 21))
                sub_val_list = list(range(6, 11))
            elif split == 0:
                sub_list = list(range(6, 21))
                sub_val_list = list(range(1, 6))

        else:
            if use_split_coco:
                print('INFO: using SPLIT COCO')
                class_list = list(range(1, 81))
                if split == 3:
                    sub_val_list = list(range(4, 81, 4))
                    sub_list = list(set(class_list) - set(sub_val_list))                    
                elif split == 2:
                    sub_val_list = list(range(3, 80, 4))
                    sub_list = list(set(class_list) - set(sub_val_list))    
                elif split == 1:
                    sub_val_list = list(range(2, 79, 4))
                    sub_list = list(set(class_list) - set(sub_val_list))    
                elif split == 0:
                    sub_val_list = list(range(1, 78, 4))
                    sub_list = list(set(class_list) - set(sub_val_list))    
            else:
                print('INFO: using COCO')
                class_list = list(range(1, 81))
                if split == 3:
                    sub_list = list(range(1, 61))
                    sub_val_list = list(range(61, 81))
                elif split == 2:
                    sub_list = list(range(1, 41)) + list(range(61, 81))
                    sub_val_list = list(range(41, 61))
                elif split == 1:
                    sub_list = list(range(1, 21)) + list(range(41, 81))
                    sub_val_list = list(range(21, 41))
                elif split == 0:
                    sub_list = list(range(21, 81)) 
                    sub_val_list = list(range(1, 21))    
                    

        # Processing data
        self.sub_list, self.sub_val_list = sub_list, sub_val_list
        sub_list = sub_list if mode == 'train' else sub_val_list

        if not os.path.isfile(data_list):
            raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

        print("Processing data...")
        
        image_label_list = []
        sub_class_file_list = defaultdict(list)

        with open(data_list) as file:
            list_read = file.readlines()
            
            for l_idx in tqdm(range(len(list_read))):
                line_split = list_read[l_idx].strip().split()

                image_name = os.path.join(data_root, line_split[0])
                label_name = os.path.join(data_root, line_split[1])
                
                item = (image_name, label_name)

                if not os.path.isfile(label_name):
                    continue
                
                if mode == 'train':
                    label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
                elif mode == 'val':
                    label = np.array(Image.open(label_name))                    

                label_class = np.unique(label).tolist()
                label_class = list(set(label_class) - {0, 255})

                # removing samples with small masks
                filtered_label_class = []       
                for c in label_class:
                    if c not in sub_list:
                        continue

                    target_pix_count = np.sum(label == c)
                    if target_pix_count >= 2 * 32 * 32:
                        filtered_label_class.append(c)


                label_class = filtered_label_class    

                if len(label_class) > 0:
                    image_label_list.append(item)
                    for c in label_class:
                        if c in sub_list:
                            sub_class_file_list[c].append(item)

        
        print("Checking image&label pair {} list done! ".format(split))

        self.dataset_samples, self.sub_class_file_list = (
            image_label_list, 
            sub_class_file_list
        )

    def get_raw_image_label(self, image_path, label_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        if self.mode == 'train':
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        elif self.mode == 'val':
            label = np.array(Image.open(label_path))   
            
        assert image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1], \
                "Image & label shape mismatch"

        return image, label
    
    def choose_label(self, label, class_chosen):
        target_pix = np.where(label == class_chosen)
        ignore_pix = np.where(label == 255)
        
        chosen_label = np.zeros_like(label)
        if target_pix[0].shape[0] > 0:
            chosen_label[target_pix[0], target_pix[1]] = 1 
        chosen_label[ignore_pix[0], ignore_pix[1]] = 255

        return chosen_label

