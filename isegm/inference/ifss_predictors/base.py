import torch
import torch.nn.functional as F
from torchvision import transforms
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide


class BasePredictor(object):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 **kwargs):
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        
        self.original_query_image = None
        self.original_support_image = None
        
        self.device = device
        self.zoom_in = zoom_in
        
        self.prev_query_prediction = None
        self.prev_support_prediction = None
        
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

        # TODO: add support for these
        # self.transforms = [zoom_in] if zoom_in is not None else []
        self.transforms = []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.transforms.append(SigmoidForPred())
        
        # TODO: add support for these
        # if with_flip:
        #     self.transforms.append(AddHorizontalFlip())

    def set_input_images(self, query_image, support_image):
        query_image_nd = self.to_tensor(query_image)
        for transform in self.transforms:
            transform.reset()
        self.original_query_image = query_image_nd.to(self.device)
        if len(self.original_query_image.shape) == 3:
            self.original_query_image = self.original_query_image.unsqueeze(0)
        self.prev_query_prediction = torch.zeros_like(self.original_query_image[:, :1, :, :])
        
        
        support_image_nd = self.to_tensor(support_image)
        for transform in self.transforms:
            transform.reset()
        self.original_support_image = support_image_nd.to(self.device)
        if len(self.original_support_image.shape) == 3:
            self.original_support_image = self.original_support_image.unsqueeze(0)
        self.prev_support_prediction = torch.zeros_like(self.original_support_image[:, :1, :, :])

    def get_prediction(self, clicker, prev_query_mask=None, prev_support_mask=None):
        clicks_list = clicker.get_clicks()

        # gets right model when needed 
        if self.click_models is not None:
            model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        # prep query image
        input_query_image = self.original_query_image
        if prev_query_mask is None:
            prev_query_mask = self.prev_query_prediction
        query_image_nd, _, _ = self.apply_transforms(  # BUG: no idea what this is doing
            input_query_image, [clicks_list]
        )

        # prep support image
        input_support_image = self.original_support_image
        if prev_support_mask is None:
            prev_support_mask = self.prev_support_prediction
        support_image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_support_image, [clicks_list]
        )

        query_pred_logits, support_pred_logits = self._get_prediction(
            support_image_nd, 
            prev_support_mask, 
            clicks_lists, 
            query_image_nd, 
            prev_query_mask, 
            is_image_changed
        )
        
        query_prediction = F.interpolate(
            query_pred_logits, 
            mode='bilinear', 
            align_corners=True,
            size=query_pred_logits.size()[2:]
        )
        
        support_prediction = F.interpolate(
            support_pred_logits, 
            mode='bilinear', 
            align_corners=True,
            size=support_pred_logits.size()[2:]
        )

        for t in reversed(self.transforms):
            query_prediction = t.inv_transform(query_prediction)
            support_prediction = t.inv_transform(support_prediction)

        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
            return self.get_prediction(clicker)

        self.prev_query_prediction = query_prediction
        self.prev_support_prediction = support_prediction
        return query_prediction.cpu().numpy()[0, 0], support_prediction.cpu().numpy()[0, 0]

    def _get_prediction(
        self, 
        support_image_nd, 
        prev_support_mask, 
        clicks_lists, 
        query_image_nd, 
        prev_query_mask, 
        is_image_changed
    ):
        points_nd = self.get_points_nd(clicks_lists)
        outputs = self.net(
            support_image_nd, 
            prev_support_mask, 
            points_nd, 
            query_image_nd, 
            prev_query_mask, 
        )
        
        return outputs['q_masks'], outputs['s_instances']

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']
