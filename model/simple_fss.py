from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SimpleClick.isegm.model.modeling.models_vit import VisionTransformer
from model.SimpleClick.isegm.model.is_plainvit_model import SimpleFPN
from model.SimpleClick.isegm.model.modeling.swin_transformer import (
    SwinTransfomerSegHead,
)


class FewShotViTModel(nn.Module):
    """
    Supports only one shot learning
    """

    def __init__(
        self,
        backbone_params={},
        neck_params={},
        head_params={},
        random_split=False,
        backbone_lr_mult=0.0,
        shared_neck=False,
    ):
        super().__init__()
        self.random_split = random_split
        self.backbone_lr_mult = backbone_lr_mult
        self.shared_neck = shared_neck

        # Shared backbone and neck
        self.backbone = VisionTransformer(**backbone_params)
        if shared_neck:
            self.neck = SimpleFPN(**neck_params)
        else:
            self.support_neck = SimpleFPN(**neck_params)
            self.query_neck = SimpleFPN(**neck_params)

        self.query_head = SwinTransfomerSegHead(**head_params)

        self.backbone.init_weights_from_pretrained(
            "./weights/mae_pretrain_vit_base.pth"
        )
        for param in self.backbone.parameters():
            if self.backbone_lr_mult == 0.0:
                param.requires_grad = False
            else:
                param.lr_mult = self.backbone_lr_mult

        # --- PFENet stuff ---
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, x, s_x, s_y, y):
        """
        This is a temporary wraper for PFENet framework
        """
        out = self._forward(
            edict(
                support=edict(image=s_x[:, 0], mask=s_y), query=edict(image=x, mask=y)
            )
        )
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=True)
        if self.training:
            main_loss = self.criterion(out, y.long())
            return out.max(1)[1], main_loss, torch.tensor(0.0)
        else:
            return out

    def _forward(self, input):
        with torch.set_grad_enabled(self.backbone_lr_mult != 0.0):
            images = torch.cat([input.support.image, input.query.image], dim=0)
            multi_scale_features = self.get_features(images)

        multi_scale_fused_features = []  # goes into the head
        for feature in (
            multi_scale_features if self.shared_neck else zip(*multi_scale_features)
        ):
            support_feature, query_feature = (
                feature.chunk(2, dim=0) if self.shared_neck else feature
            )

            rescaled_mask = F.interpolate(
                input.support.mask.float(),
                size=(support_feature.size(2), support_feature.size(3)),
                mode="bilinear",
                align_corners=True,
            )
            prototype = self.weighted_gap(support_feature, rescaled_mask)
            expanded_prototype = prototype.expand_as(query_feature)
            multi_scale_fused_features.append(
                torch.cat([query_feature, expanded_prototype], dim=1)
            )

        return self.query_head(multi_scale_fused_features)

    def get_features(self, image):
        """
        Args:
            image: [B, 3, H, W]

        Returns:
            [
                torch.Size(B, self.neck.out_dims[0], H/4, W/4),
                torch.Size(B, self.neck.out_dims[0], H/8, W/8),
                torch.Size(B, self.neck.out_dims[0], H/16, W/16),
                torch.Size(B, self.neck.out_dims[0], H/32, W/32),
            ]

        e.g. [B, 3, 224, 224] -> [
                [B, 256, 56, 56],
                [B, 256, 28, 28],
                [B, 256, 14, 14],
                [B, 256, 7, 7]
            ]

        """
        backbone_features = self.backbone.forward_backbone(
            image, shuffle=self.random_split
        )

        # Extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape
        grid_size = self.backbone.patch_embed.grid_size

        backbone_features = backbone_features.transpose(-1, -2).view(
            B, C, grid_size[0], grid_size[1]
        )
        # return self.neck(backbone_features)
        support_features, query_features = backbone_features.chunk(2, dim=0)
        support_features = self.support_neck(support_features)
        query_features = self.query_neck(query_features)

        return [support_features, query_features]

    def weighted_gap(self, feature, mask):
        """
        Args:
            feature: [B, f_c, f_h, f_w]
            mask: [B, 1, H, W]

        Returns:
            [B, f_c, 1, 1]
        """
        feature = feature * mask
        feat_h = feature.shape[-2:][0]
        feat_w = feature.shape[-2:][1]
        area = (
            F.avg_pool2d(mask, (feature.size()[2], feature.size()[3])) * feat_h * feat_w
            + 0.0005
        )
        feature = (
            F.avg_pool2d(input=feature, kernel_size=feature.shape[-2:])
            * feat_h
            * feat_w
            / area
        )
        return feature
