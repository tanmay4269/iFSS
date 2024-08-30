import torch
import torch.nn as nn
import torch.nn.functional as F

from isegm.utils.serialization import serialize
from .ifss_model import iFSSModel
from .modeling.hrnet_ocr import HighResolutionNet
from isegm.model.modifiers import LRMult


class CustomHRNet(HighResolutionNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x, additional_features=None):
        x, feats_list = self.encoder(x, additional_features)
        output = self.decoder(x)
        
        return output, feats_list
    
    def encoder(self, x, additional_features):
        features_list = self.compute_hrnet_feats(x, additional_features)
        
        x = self.aggregate_hrnet_features(features_list)
        return x, features_list
        
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
        
    def compute_hrnet_feats(self, x, additional_features=None, do_till_stage=4):
        x = self.compute_pre_stage_features(x, additional_features)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        
        if do_till_stage == 2:
            return y_list

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
        
        if do_till_stage == 3:
            return y_list

        x_list = []
        for i in range(self.stage4_num_branches):
            if self.transition3[i] is not None:
                if i < self.stage3_num_branches:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        return x
    
    def do_stage_4(self, y_list):
        x_list = []
        for i in range(self.stage4_num_branches):
            if self.transition3[i] is not None:
                if i < self.stage3_num_branches:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        
        return x
        

class HRNetModel(iFSSModel):
    @serialize
    def __init__(self, 
            width=48, 
            ocr_width=256, 
            small=False, 
            backbone_lr_mult=0.1,
            norm_layer=nn.BatchNorm2d, 
            mid_feats_idx=2,
            **kwargs
    ):
        super().__init__(norm_layer=norm_layer, **kwargs)
        
        self.mid_feats_idx = mid_feats_idx  # TODO: code isnt compatable with other values than 2

        self.support_net = CustomHRNet(
            width=width, 
            ocr_width=ocr_width, 
            small=small,
            num_classes=1, 
            norm_layer=norm_layer
        )
        self.support_net.apply(LRMult(backbone_lr_mult))
        if ocr_width > 0:
            self.support_net.ocr_distri_head.apply(LRMult(1.0))
            self.support_net.ocr_gather_head.apply(LRMult(1.0))
            self.support_net.conv3x3_ocr.apply(LRMult(1.0))
            
        self.query_net = CustomHRNet(
            width=width, 
            ocr_width=ocr_width, 
            small=small,
            num_classes=1, 
            norm_layer=norm_layer
        )
        
        # shrinking query input
        self.query_input = nn.Conv2d(4, 3, 1)
        
        # shrinking dimentions of fused query and support features
        self.conv1 = nn.Conv2d(72 + 72, 72, 1)  # :')

    def support_forward(self, image, coord_features=None):
        outputs, feats = self.support_net(image, coord_features)
        
        s_pred = (torch.sigmoid(outputs[0]) > 0.5).int()
        feats = feats[self.mid_feats_idx]

        _, _, H, W = feats.shape
        pred = F.interpolate(
            s_pred.float(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        prototype = torch.sum(feats * pred, dim=(2, 3))
        
        return {
            'instances': outputs[0], 
            'instances_aux': outputs[1],
            'prototype': prototype
        }

    def query_forward(self, image, prev_output, prototype):
        x = torch.cat((prev_output, image), dim=1)
        x = self.query_input(x)
        backbone_feats = self.query_net.compute_hrnet_feats(
            x, do_till_stage=self.mid_feats_idx+1
        )
        
        feats = backbone_feats[-1]
        _, _, H, W = feats.shape
        prototype = prototype.unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)
        
        feats = torch.cat((feats, prototype), dim=1)
        
        # decoder only eats tensors of shape (B, 270, 80, 120)
        # the initial middle layer had shape (B, 72, 20, 30)
        # prototype should be of the same shape
        # I have these options now:
        #   1. naively use 1x1 conv layer to reduce the channels by half
        #       - this should be good for 1st prototype!
        
        feats = self.conv1(feats)
        
        backbone_feats[-1] = feats
        # TODO: this method is so ugly, 
        # but its prototyping 
        #                  ~~~~~~~~~~
        x = self.query_net.do_stage_4(backbone_feats)  
        x = self.query_net.aggregate_hrnet_features(x)
        
        outputs = self.query_net.decoder(x)        

        return {'masks': outputs[0], 'masks_aux': outputs[1]}