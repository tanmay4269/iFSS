import os
from time import time

import numpy as np
import torch

from isegm.inference import utils
from isegm.inference.clicker import Clicker

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        query_sample, support_sample = dataset.get_sample(index)

        sample_ious = evaluate_sample(
            query_sample.image, query_sample._encoded_masks[..., 0], 
            support_sample.image, support_sample.gt_mask, 
            predictor, sample_id=index, **kwargs)
        
        all_ious.append(sample_ious)
        
        if int(os.environ["DEBUG"]) > 0:
            if index > 2: break
        
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time


def evaluate_sample(
    query_image, query_gt_mask, 
    support_image, support_gt_mask, 
    predictor, max_iou_thr,
    pred_thr=0.49, min_clicks=1, max_clicks=20,
    sample_id=None, callback=None
):
    """
    predictor should take support and query inputs and spit out two predictions 
    """
    
    clicker = Clicker(gt_mask=support_gt_mask)
    query_pred_mask = np.zeros_like(query_gt_mask)
    support_pred_mask = np.zeros_like(support_gt_mask)
    ious_list = []

    with torch.no_grad():
        predictor.set_input_images(query_image, support_image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(support_pred_mask)
            query_pred_probs, support_pred_probs = predictor.get_prediction(clicker)
            query_pred_mask = query_pred_probs > pred_thr
            support_pred_mask = support_pred_probs > pred_thr

            q_iou = utils.get_iou(query_gt_mask, query_pred_mask)
            s_iou = utils.get_iou(support_gt_mask, support_pred_mask)
            ious_list.append((q_iou, s_iou))
            
            if callback is not None:
                callback(
                    query_image, query_gt_mask, query_pred_probs,
                    support_image, support_gt_mask, support_pred_probs, 
                    sample_id, click_indx, clicker.clicks_list
                )

            if s_iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        return np.array(ious_list, dtype=np.float32)
