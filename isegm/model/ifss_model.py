import torch
import torch.nn as nn
import numpy as np

from isegm.model.ops import DistMaps, ScaleLayer, BatchImageNormalize
from isegm.model.modifiers import LRMult


class iFSSModel(nn.Module):
    def __init__(
        self,
        use_rgb_conv=True,
        with_aux_output=False,
        norm_radius=260,
        use_disks=False,
        cpu_dist_maps=False,
        clicks_groups=None,
        with_prev_mask=False,
        use_leaky_relu=False,
        binary_prev_mask=False,
        conv_extend=False,
        norm_layer=nn.BatchNorm2d,
        norm_mean_std=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ):
        super().__init__()
        self.with_aux_output = with_aux_output
        self.clicks_groups = clicks_groups
        self.with_prev_mask = with_prev_mask
        self.binary_prev_mask = binary_prev_mask
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])

        self.coord_feature_ch = 2
        if clicks_groups is not None:
            self.coord_feature_ch *= len(clicks_groups)

        if self.with_prev_mask:
            self.coord_feature_ch += 1

        if use_rgb_conv:
            rgb_conv_layers = [
                nn.Conv2d(
                    in_channels=3 + self.coord_feature_ch,
                    out_channels=6 + self.coord_feature_ch,
                    kernel_size=1,
                ),
                norm_layer(6 + self.coord_feature_ch),
                (
                    nn.LeakyReLU(negative_slope=0.2)
                    if use_leaky_relu
                    else nn.ReLU(inplace=True)
                ),
                nn.Conv2d(
                    in_channels=6 + self.coord_feature_ch, out_channels=3, kernel_size=1
                ),
            ]
            self.rgb_conv = nn.Sequential(*rgb_conv_layers)
        elif conv_extend:
            self.rgb_conv = None
            self.maps_transform = nn.Conv2d(
                in_channels=self.coord_feature_ch,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            self.maps_transform.apply(LRMult(0.1))
        else:
            self.rgb_conv = None
            mt_layers = [
                nn.Conv2d(
                    in_channels=self.coord_feature_ch, out_channels=16, kernel_size=1
                ),
                (
                    nn.LeakyReLU(negative_slope=0.2)
                    if use_leaky_relu
                    else nn.ReLU(inplace=True)
                ),
                nn.Conv2d(
                    in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1
                ),
                ScaleLayer(init_value=0.05, lr_mult=1),
            ]
            self.maps_transform = nn.Sequential(*mt_layers)

        if self.clicks_groups is not None:
            self.dist_maps = nn.ModuleList()
            for click_radius in self.clicks_groups:
                self.dist_maps.append(
                    DistMaps(
                        norm_radius=click_radius,
                        spatial_scale=1.0,
                        cpu_mode=cpu_dist_maps,
                        use_disks=use_disks,
                    )
                )
        else:
            self.dist_maps = DistMaps(
                norm_radius=norm_radius,
                spatial_scale=1.0,
                cpu_mode=cpu_dist_maps,
                use_disks=use_disks,
            )

    def forward(self, s_inputs, q_inputs, pretraining_enabled):
        """
        Args: 
            - s_inputs (dict):
                - image
                - gt
                - prev_output
                - points
            - q_inputs (dict):
                - image
                - gt
                - prev_output

        Returns:
            - support and query instances, masks and their auxilaries

        TODO: 
            - [ ] Make `pretraining_enabled` a class attribute 
        """

        s_image, prev_s_mask = self.prepare_input(s_inputs.image, s_inputs.prev_output)
        coord_features = self.get_coord_features(s_image, prev_s_mask, s_inputs.points)

        if self.rgb_conv is not None:
            x = self.rgb_conv(torch.cat((s_image, coord_features), dim=1))
            # s_outputs = self.support_forward(
            #     x, s_inputs.gt if pretraining_enabled else None)
            s_outputs = self.support_forward(x, s_inputs.gt)  # ! Temporary fix
        else:
            coord_features = self.maps_transform(coord_features)
            # s_outputs = self.support_forward(
            #     s_image, 
            #     s_inputs.gt if pretraining_enabled else None, 
            #     coord_features)
            s_outputs = self.support_forward(s_image, s_inputs.gt, coord_features)  # ! Temporary fix

        if not pretraining_enabled:
            helpers = s_outputs.pop("query_helpers", None)
            if helpers is None:  # For hrnet-ifss version
                helpers = s_outputs["prototypes"]
            else:
                helpers["q_gt"] = q_inputs.gt
            q_outputs = self.query_forward(
                q_inputs.image,
                q_inputs.prev_output,
                helpers,
            )

        s_outputs["instances"] = nn.functional.interpolate(
            s_outputs["instances"],
            size=s_image.size()[2:],
            mode="bilinear",
            align_corners=True,
        )

        if not pretraining_enabled:
            q_outputs["masks"] = nn.functional.interpolate(
                q_outputs["masks"],
                size=q_inputs.image.size()[2:],
                mode="bilinear",
                align_corners=True,
            )

        # Removed for PFENet-iFSS
        # if self.with_aux_output:
        #     s_outputs["instances_aux"] = nn.functional.interpolate(
        #         s_outputs["instances_aux"],
        #         size=s_image.size()[2:],
        #         mode="bilinear",
        #         align_corners=True,
        #     )

        #     if not pretraining_enabled:
        #         q_outputs["masks_aux"] = nn.functional.interpolate(
        #             q_outputs["masks_aux"],
        #             size=q_inputs.image.size()[2:],
        #             mode="bilinear",
        #             align_corners=True,
        #         )

        outputs = {}
        for k, v in s_outputs.items():
            if k == "prototype":
                continue
            outputs[f"s_{k}"] = v

        if not pretraining_enabled:
            for k, v in q_outputs.items():
                outputs[f"q_{k}"] = v

        return outputs

    def prepare_input(self, image, prev_mask):
        if self.binary_prev_mask:
            prev_mask = (prev_mask > 0.5).float()

        # image = self.normalization(image) # ! Doesn't work for pfenet-ritm
                                            # ! no idea why i put it in the first place
                                            # ! even for ifss-ritm, the dataset already
                                            # ! normalizes the image
        return image, prev_mask

    def backbone_forward(self, image, coord_features=None):
        raise NotImplementedError

    def get_coord_features(self, image, prev_mask, points):
        if self.clicks_groups is not None:
            points_groups = split_points_by_order(
                points, groups=(2,) + (1,) * (len(self.clicks_groups) - 2) + (-1,)
            )
            coord_features = [
                dist_map(image, pg)
                for dist_map, pg in zip(self.dist_maps, points_groups)
            ]
            coord_features = torch.cat(coord_features, dim=1)
        else:
            coord_features = self.dist_maps(image, points)

        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)

        return coord_features


def split_points_by_order(tpoints: torch.Tensor, groups):
    points = tpoints.cpu().numpy()
    num_groups = len(groups)
    bs = points.shape[0]
    num_points = points.shape[1] // 2

    groups = [x if x > 0 else num_points for x in groups]
    group_points = [np.full((bs, 2 * x, 3), -1, dtype=np.float32) for x in groups]

    last_point_indx_group = np.zeros((bs, num_groups, 2), dtype=np.int)
    for group_indx, group_size in enumerate(groups):
        last_point_indx_group[:, group_indx, 1] = group_size

    for bindx in range(bs):
        for pindx in range(2 * num_points):
            point = points[bindx, pindx, :]
            group_id = int(point[2])
            if group_id < 0:
                continue

            is_negative = int(pindx >= num_points)
            if group_id >= num_groups or (
                group_id == 0 and is_negative
            ):  # disable negative first click
                group_id = num_groups - 1

            new_point_indx = last_point_indx_group[bindx, group_id, is_negative]
            last_point_indx_group[bindx, group_id, is_negative] += 1

            group_points[group_id][bindx, new_point_indx, :] = point

    group_points = [
        torch.tensor(x, dtype=tpoints.dtype, device=tpoints.device)
        for x in group_points
    ]

    return group_points
