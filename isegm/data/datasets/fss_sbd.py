import os
import json
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import cv2
import torch
import numpy as np
from PIL import Image
from scipy.io import loadmat

from isegm.utils.misc import get_bbox_from_mask, get_labels_with_sizes
from isegm.data.ifss_base import iFSSDataset
from isegm.data.sample import DSample


class iFSS_SBD_Dataset(iFSSDataset):
    def __init__(
        self,
        cfg,
        data_root,
        data_list,
        mode="train",
        split=0,
        supports=1,
        queries=1,
        use_coco=False,
        use_split_coco=False,
        buggy_mask_thresh=0.08,
        **kwargs,
    ):
        super(iFSS_SBD_Dataset, self).__init__(**kwargs)
        
        # FIXME: Find a better way of doing this
        self.image_root = cfg.SBD_IMAGE_PATH
        self.label_root = cfg.SBD_LABEL_PATH

        self.cfg = cfg
        self.mode = mode
        self.supports = supports
        self.queries = queries

        self.data_root = Path(data_root)
        self._buggy_objects = dict()
        self._buggy_mask_thresh = buggy_mask_thresh

        # TODO: write this better, this implementation is probably not standard
        self.dataset_samples, self.sub_class_file_list = self.make_dataset(
            mode,
            use_coco,
            use_split_coco,
            split,
            data_root,
            data_list,
        )

        if self.cfg.debug == "one_batch_overfit":
            self.dataset_samples = self.dataset_samples[:cfg.batch_size]
            for k, v in self.sub_class_file_list.items():
                self.sub_class_file_list[k] = v[:cfg.batch_size]

        ...
        # ! Doesn't work now :(
        # # For consistent validation
        # if mode != "val":
        #     return

        # val_samples_filename = (
        #     f"val_samples_split{split}_{np.random.randint(10, 100)}.pt"
        # )
        # val_samples_path = os.path.join(cfg.SBD_CACHE, val_samples_filename)
        # if False and val_samples_filename in os.listdir(cfg.SBD_CACHE):
        #     print("Loading val samples from cache...", end="")
        #     self.val_samples = torch.load(val_samples_path)
        #     print("Done!")
        # else:
        #     print("Saving val samples...")
        #     self.val_samples = []
        #     for i in tqdm(range(len(self.dataset_samples)), desc="Processing samples"):
        #         self.val_samples.append(self.get_sample(i))
        #     print("Saving samples to cache...", end="")
        #     torch.save(self.val_samples, val_samples_path)
        #     print("Done!")
        #     ...
        # self.mode = "val-loaded"

    def get_sample(self, index):
        if self.cfg.debug == "one_batch_overfit":
            np.random.seed(42)
        # if self.mode == "val-loaded":
        #     return self.val_samples[index]

        image_path, label_path = self.dataset_samples[index]
        image, label = self.get_raw_image_label(image_path, label_path)

        label_class = np.unique(label).tolist()
        label_class = list(set(label_class) - {0, 255})

        filtered_label_class = []
        for c in label_class:
            if c in self.sub_val_list and self.mode in ["val", "test"]:
                filtered_label_class.append(c)
            if c in self.sub_list and self.mode == "train":
                filtered_label_class.append(c)
        label_class = filtered_label_class
        assert len(label_class) > 0

        # making a new query label
        class_chosen = np.random.choice(label_class)
        label = self.choose_label(label, class_chosen)

        # selecting support pair
        file_class_chosen = self.sub_class_file_list[class_chosen]
        num_files = len(file_class_chosen)

        while True:
            support_idx = np.random.randint(0, num_files - 1)
            support_image_path, support_label_path = file_class_chosen[support_idx]

            if support_image_path != image_path or support_label_path != label_path:
                break

        support_image, support_label = self.get_raw_image_label(
            support_image_path, support_label_path
        )
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
                image,
                query_instances_mask,
                objects_ids=[query_instances_ids[0]],
                sample_id=index,
            ),
            # Support
            DSample(
                support_image,
                support_instances_mask,
                objects_ids=[support_instances_ids[0]],
                sample_id=support_idx,
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
        mode="train",
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
                print("INFO: using SPLIT COCO")
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
                print("INFO: using COCO")
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
        sub_list = sub_list if mode == "train" else sub_val_list

        if not os.path.isfile(data_list):
            raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

        dump_file_name = f"split{split}_{mode}.json"
        if dump_file_name in os.listdir(self.cfg.SBD_CACHE):
            print("Loading from saved dump")
            with open(os.path.join(self.cfg.SBD_CACHE, dump_file_name), "r") as file:
                loaded_data = json.load(file)

            sub_class_file_list = defaultdict(list)
            for k, v in loaded_data[1].items():
                sub_class_file_list[int(k)] = v

            return loaded_data[0], sub_class_file_list

        print(f"Couldn't find {dump_file_name}\tProcessing data...", end="")

        image_label_list = []
        sub_class_file_list = defaultdict(list)

        with open(data_list) as file:
            list_read = file.readlines()

            for l_idx in tqdm(range(len(list_read))):
                line_split = list_read[l_idx].strip().split()

                # image_name = os.path.join(data_root, line_split[0])
                # label_name = os.path.join(data_root, line_split[1])
                
                image_name = os.path.join(self.image_root, line_split[0]) + '.jpg'
                label_name = os.path.join(self.label_root, line_split[0]) + '.mat'

                item = (image_name, label_name)

                if not os.path.isfile(label_name):
                    continue

                # if mode == "train":
                #     label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
                # elif mode == "val":
                #     label = np.array(Image.open(label_name))
                
                label = loadmat(label_name)
                label = np.array(label["GTinst"][0]["Segmentation"][0], dtype=np.uint8)

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
        print("Done!")

        print(f"Dumping to {dump_file_name}...", end="")
        with open(os.path.join(self.cfg.SBD_CACHE, dump_file_name), "w") as file:
            data = (image_label_list, sub_class_file_list)
            json.dump(data, file, indent=None)
        print("Done!")

        return image_label_list, sub_class_file_list

    def get_raw_image_label(self, image_path, label_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # if self.mode == "train":
        #     label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        # elif self.mode == "val":
        #     label = np.array(Image.open(label_path))
            
        label = loadmat(label_path)
        label = np.array(label["GTinst"][0]["Segmentation"][0], dtype=np.uint8)

        assert (
            image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1]
        ), "Image & label shape mismatch"

        return image, label

    def choose_label(self, label, class_chosen):
        target_pix = np.where(label == class_chosen)
        ignore_pix = np.where(label == 255)

        chosen_label = np.zeros_like(label)
        if target_pix[0].shape[0] > 0:
            chosen_label[target_pix[0], target_pix[1]] = 1
        chosen_label[ignore_pix[0], ignore_pix[1]] = 255

        return chosen_label
