from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.nn.functional as F

from isegm.utils.serialization import serialize
from .ifss_model import iFSSModel
from .modeling.hrnet_ocr import HighResolutionNet
from isegm.model.modifiers import LRMult


class CustomHRNet(HighResolutionNet):
    """
    Customization:
        - Added `query_encoder` method that fuses support prototypes
    """

    def __init__(self, is_query_net=False, **kwargs):
        super().__init__(**kwargs)

        if is_query_net:
            self.conv1x1 = nn.ModuleList([
                nn.Conv2d(256 + 18, 256, 1),
                nn.Conv2d(36 + 36, 36, 1),
                nn.Conv2d(72 + 72, 72, 1),
                nn.Conv2d(144 + 144, 144, 1),
            ])

    def forward(self, x, additional_features=None):
        x, feats_list = self.encoder(x, additional_features)
        output = self.decoder(x)

        return output, feats_list

    def encoder(self, x, additional_features):
        return self.compute_hrnet_feats(x, additional_features)

    def decoder(self, feats):
        if self.ocr_width > 0:
            out_aux = self.aux_head(feats)
            feats = self.conv3x3_ocr(feats)

            context = self.ocr_gather_head(feats, out_aux)
            feats = self.ocr_distri_head(feats, context)
            out = self.cls_head(feats)
            return [out, out_aux]
        else:
            return [self.cls_head(feats), None]

    def aggregate_hrnet_features(self, x):
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(
            x[1], size=(x0_h, x0_w), mode="bilinear", align_corners=self.align_corners
        )
        x2 = F.interpolate(
            x[2], size=(x0_h, x0_w), mode="bilinear", align_corners=self.align_corners
        )
        x3 = F.interpolate(
            x[3], size=(x0_h, x0_w), mode="bilinear", align_corners=self.align_corners
        )

        return torch.cat([x[0], x1, x2, x3], 1), x

    def query_encoder(self, x, prototypes):
        """
        Concatinates support prototype at each layer and reduces
            dimentionality using conv1x1.
        """
        x = self.compute_pre_stage_features(
            x, additional_features=None
        )  # (64, 80, 120)
        x = self.layer1(x)  # (256, 80, 120)

        _, _, H, W = x.shape
        prototype = prototypes[0].unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)
        x = torch.cat((x, prototype), dim=1)
        x = self.conv1x1[0](x)

        x_list = []
        for i in range(self.stage2_num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        # (18, 80, 120)
        # (36, 40, 60)

        _, _, H, W = y_list[-1].shape
        prototype = prototypes[1].unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)
        x = torch.cat((y_list[-1], prototype), dim=1)
        y_list[-1] = self.conv1x1[1](x)

        x_list = []
        for i in range(self.stage3_num_branches):
            if self.transition2[i] is not None:
                if i < self.stage2_num_branches:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        # (18, 80, 120)
        # (36, 40, 60)
        # (72, 20, 30)

        _, _, H, W = y_list[-1].shape
        prototype = prototypes[2].unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)
        x = torch.cat((y_list[-1], prototype), dim=1)
        y_list[-1] = self.conv1x1[2](x)

        x_list = []
        for i in range(self.stage4_num_branches):
            if self.transition3[i] is not None:
                if i < self.stage3_num_branches:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        # (18, 80, 120)
        # (36, 40, 60)
        # (72, 20, 30)
        # (144, 10, 15)

        _, _, H, W = y_list[-1].shape
        prototype = prototypes[3].unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)
        x = torch.cat((y_list[-1], prototype), dim=1)
        y_list[-1] = self.conv1x1[3](x)

        out, _ = self.aggregate_hrnet_features(y_list)

        return out


class iFSS_HRNetModel(iFSSModel):
    @serialize
    def __init__(
        self,
        width=48,
        ocr_width=256,
        small=False,
        lr_mult=edict(
            support = edict(
                default = 1.0,
                backbone = 0.1
            ),
            query = edict(
                default = 1.0,
                backbone = 0.1
            )
        ),
        norm_layer=nn.BatchNorm2d,
        **kwargs
    ):
        super().__init__(norm_layer=norm_layer, **kwargs)

        # ===== Support Branch =====
        self.support_net = CustomHRNet(
            width=width,
            ocr_width=ocr_width,
            small=small,
            num_classes=1,
            norm_layer=norm_layer,
        )

        self.support_net.apply(LRMult(lr_mult.support.backbone))
        if ocr_width > 0:
            lr = lr_mult.support.default
            self.support_net.ocr_distri_head.apply(LRMult(lr))
            self.support_net.ocr_gather_head.apply(LRMult(lr))
            self.support_net.conv3x3_ocr.apply(LRMult(lr))

        # ===== Query Branch =====
        use_leaky_relu = True
        query_input_layers = [
            nn.Conv2d(in_channels=3 + 1, out_channels=6 + 1, kernel_size=1),
            norm_layer(6 + 1),
            (
                nn.LeakyReLU(negative_slope=0.2)
                if use_leaky_relu
                else nn.ReLU(inplace=True)
            ),
            nn.Conv2d(in_channels=6 + 1, out_channels=3, kernel_size=1),
        ]

        self.query_input = nn.Sequential(*query_input_layers)

        self.query_net = CustomHRNet(
            is_query_net=True,
            width=width,
            ocr_width=ocr_width,
            small=small,
            num_classes=1,
            norm_layer=norm_layer,
        )

        self.query_net.apply(LRMult(lr_mult.query.backbone))
        lr = lr_mult.query.default
        self.query_net.conv1x1.apply(LRMult(lr))
        if ocr_width > 0:
            self.query_net.ocr_distri_head.apply(LRMult(lr))
            self.query_net.ocr_gather_head.apply(LRMult(lr))
            self.query_net.conv3x3_ocr.apply(LRMult(lr))


    def support_forward(
        self, image, s_gt=None, coord_features=None, filter_threshold=0.5
    ):
        """
        Args:
            - `s_gt` is available during pretraining
        """

        outputs, feature_list = self.support_net(image, coord_features)

        s_filter = (
            (torch.sigmoid(outputs[0]) > filter_threshold).float()
            if s_gt is None
            else s_gt
        )

        prototypes = []
        for features in feature_list:
            pred = F.interpolate(
                s_filter,
                size=features.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

            prototype = torch.mean(features * pred, dim=(2, 3))
            prototypes.append(prototype)

        return {
            "instances": outputs[0],
            "instances_aux": outputs[1],
            "prototypes": prototypes,
        }

    def query_forward(self, image, prev_output, prototypes):
        prev_output = torch.sigmoid(prev_output)
        x = self.query_input(torch.cat((image, prev_output), dim=1))
        x = self.query_net.query_encoder(x, prototypes)
        outputs = self.query_net.decoder(x)

        return {"masks": outputs[0], "masks_aux": outputs[1]}
