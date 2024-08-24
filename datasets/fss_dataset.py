import os
import os.path
import random

import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import AlbumentationsTransformWrapper


class FSSDataset(Dataset):
    def __init__(self, split=3, shot=1, data_root=None, data_list=None, transform=None, mode='train', use_coco=False, use_split_coco=False):
        assert mode in ['train', 'val', 'test']
        
        self.mode = mode
        self.split = split  
        self.shot = shot
        self.data_root = data_root   

        if not use_coco:  # Pascal and/or SBD
            self.class_list = list(range(1, 21))
            if self.split == 3: 
                self.sub_list = list(range(1, 16))
                self.sub_val_list = list(range(16, 21))
            elif self.split == 2:
                self.sub_list = list(range(1, 11)) + list(range(16, 21))
                self.sub_val_list = list(range(11, 16))
            elif self.split == 1:
                self.sub_list = list(range(1, 6)) + list(range(11, 21))
                self.sub_val_list = list(range(6, 11))
            elif self.split == 0:
                self.sub_list = list(range(6, 21))
                self.sub_val_list = list(range(1, 6))

        else:
            if use_split_coco:
                print('INFO: using SPLIT COCO')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_val_list = list(range(4, 81, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))                    
                elif self.split == 2:
                    self.sub_val_list = list(range(3, 80, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 1:
                    self.sub_val_list = list(range(2, 79, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 0:
                    self.sub_val_list = list(range(1, 78, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
            else:
                print('INFO: using COCO')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_list = list(range(1, 61))
                    self.sub_val_list = list(range(61, 81))
                elif self.split == 2:
                    self.sub_list = list(range(1, 41)) + list(range(61, 81))
                    self.sub_val_list = list(range(41, 61))
                elif self.split == 1:
                    self.sub_list = list(range(1, 21)) + list(range(41, 81))
                    self.sub_val_list = list(range(21, 41))
                elif self.split == 0:
                    self.sub_list = list(range(21, 81)) 
                    self.sub_val_list = list(range(1, 21))    

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)

        if self.mode == 'train':
            self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_list)
        elif self.mode == 'val':
            self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_val_list)
            
        self.transform = AlbumentationsTransformWrapper(transform, return_tensor=True)


    def __getitem__(self, index):
        label_class = []
        image_path, label_path = self.data_list[index]
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
        
        # selecting support pairs
        file_class_chosen = self.sub_class_file_list[class_chosen]
        num_files = len(file_class_chosen)

        support_image_list = []
        support_label_list = []
        subcls_list = []
        support_idx_list = []
        for k in range(self.shot):
            _list = self.sub_list if self.mode == 'train' else self.sub_val_list
            subcls_list.append(_list.index(class_chosen))

            while True:
                support_idx = random.randint(0, num_files - 1)
                support_image_path, support_label_path = file_class_chosen[support_idx]
                
                if (
                    (support_image_path != image_path 
                     or support_label_path != label_path) 
                    and support_idx not in support_idx_list
                ):
                    break

            support_idx_list.append(support_idx)

            support_image, support_label = self.get_raw_image_label(support_image_path, support_label_path)
            support_label = self.choose_label(support_label, class_chosen)            
            
            support_image_list.append(support_image)
            support_label_list.append(support_label)
        
        assert len(support_label_list) == self.shot and len(support_image_list) == self.shot

        raw_label = label.copy()
        if self.transform is not None:
            image, label = self.transform(image, label)

            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = \
                    self.transform(support_image_list[k], support_label_list[k])

        s_x = torch.stack(support_image_list, dim=0)
        s_y = torch.stack(support_label_list, dim=0)

        image, label, s_x, s_y = image.float() / 255.0, label.long(), s_x.float() / 255.0, s_y.long()

        return image, label, s_x, s_y
        # return image, label, s_x, s_y, subcls_list, raw_label


    def get_raw_image_label(self, image_path, label_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) 

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


    def __len__(self):
        return len(self.data_list)
    

def make_dataset(split=0, data_root=None, data_list=None, sub_list=None):    
    assert split in [0, 1, 2, 3, 10, 11, 999]
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
            
            label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)

            label_class = np.unique(label).tolist()

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
    return image_label_list, sub_class_file_list