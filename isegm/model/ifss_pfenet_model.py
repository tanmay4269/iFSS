from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.nn.functional as F

from isegm.utils.serialization import serialize
from isegm.model.modifiers import LRMult
# from .ifss_model import iFSSModel

# from .modeling.pfenet_utils import resnet as models
# from .modeling.deeplab_v3 import _DeepLabHead, _SkipProject, _ASPP
# from .modeling.basic_blocks import SepConvHead

from isegm.model.ifss_model import iFSSModel

from isegm.model.modeling.pfenet_utils import resnet as models
from isegm.model.modeling.deeplab_v3 import _DeepLabHead, _SkipProject, _ASPP
from isegm.model.modeling.basic_blocks import SepConvHead
    

class PFENetModel(iFSSModel):
    @serialize
    def __init__(
        self,
        backbone='resnet50', 
        deeplab_ch=256, 
        aspp_dropout=0.5,
        backbone_norm_layer=None, 
        backbone_lr_mult=0.1, 
        norm_layer=nn.BatchNorm2d,
        **kwargs
    ):
        super().__init__(norm_layer=norm_layer, **kwargs)
        
        self.norm_layer = norm_layer
        self.zoom_factor = 8
        self.shot = 1
        
        # Query input: encoperating previous outputs
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

        ###### Building backbone ######
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        layer0_pre_maxpool = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu1, 
            resnet.conv2, resnet.bn2, resnet.relu2, 
            resnet.conv3, resnet.bn3, resnet.relu3
        )
        self.layer0 = nn.ModuleDict({
            'pre_maxpool': layer0_pre_maxpool,
            'maxpool': resnet.maxpool
        })
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
                
        self.layer0.apply(LRMult(backbone_lr_mult))
        self.layer1.apply(LRMult(backbone_lr_mult))
        self.layer2.apply(LRMult(backbone_lr_mult))
        self.layer3.apply(LRMult(backbone_lr_mult))
        self.layer4.apply(LRMult(backbone_lr_mult))
        
        # TODO: Rename these to something more meaningful
        classes = 2
        reduce_dim = 256
        fea_dim = 1024 + 512
        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )                 

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        )  

        ppm_scales = [60, 30, 15, 8]
        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(nn.AdaptiveAvgPool2d(bin))

        mask_add_num = 1
        self.init_merge = []
        self.beta_conv = []
        self.inner_cls = []        
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim*2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))                      
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))            
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))            
        self.init_merge = nn.ModuleList(self.init_merge) 
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)                             

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )              
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        )

        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins)-1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))     
        self.alpha_conv = nn.ModuleList(self.alpha_conv)
        
        # Support decoder
        self.skip_project_in_channels = 256
        self.aspp_in_channels = 2048
        self.neck = _DeepLabHead(
            in_channels=deeplab_ch + 32, 
            mid_channels=deeplab_ch, 
            out_channels=deeplab_ch,
            norm_layer=self.norm_layer
        )
        self.skip_project = _SkipProject(
            in_channels=self.skip_project_in_channels, 
            out_channels=32, 
            norm_layer=self.norm_layer
        )
        self.aspp = _ASPP(
            in_channels=self.aspp_in_channels,
            atrous_rates=[12, 24, 36],
            out_channels=deeplab_ch,
            project_dropout=aspp_dropout,
            norm_layer=self.norm_layer
        )
        self.head = SepConvHead(
            num_outputs=1, 
            in_channels=deeplab_ch, 
            mid_channels=deeplab_ch // 2,
            num_layers=2, 
            norm_layer=norm_layer
        )
    
    def weighted_GAP(self, supp_feat, mask):
        supp_feat = supp_feat * mask
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
        return supp_feat
    
    def support_forward(
        self, image, s_gt=None, coord_features=None, filter_threshold=0.5
    ):
        """
        Args:
            - s_gt: available during pretraining (FSS pretraining)
            - coord_features: some function of the image, prev mask
                and the click points. May or may not be provided. 
        """
        
        # ! Temporary fix
        s_x = image.unsqueeze(1)
        s_y = s_gt.unsqueeze(1)
        
        # Support Feature
        # ! Improve this code: presentation and performance
        supp_feat_list = []
        final_supp_list = []
        mask_list = []
        decoder_outputs = []
        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0['pre_maxpool'](s_x[:,i,:,:,:])
                if coord_features is not None:
                    supp_feat_0 = supp_feat_0 + F.pad(
                        coord_features,
                        [0, 0, 0, 0, 0, supp_feat_0.size(1) - coord_features.size(1)],
                        mode='constant', value=0
                    )
                supp_feat_0 = self.layer0['maxpool'](supp_feat_0)
                
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                
                mask = mask[:,:,0] # Remove extra dimensions
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3 * mask)
                
                final_supp_list.append(supp_feat_4)
            
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            supp_feat = self.weighted_GAP(supp_feat, mask)
            supp_feat_list.append(supp_feat)
            
            # Decoding
            supp_feat_1 = self.skip_project(supp_feat_1)
            x = self.aspp(supp_feat_4)
            x = F.interpolate(x, supp_feat_1.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, supp_feat_1), dim=1)
            x = self.neck(x)
            x = self.head(x)
            
            x = F.interpolate(x, size=image.size()[-2:], mode='bilinear', align_corners=True)
            
            x = (torch.sigmoid(x[:, 0]) > filter_threshold).float()
            decoder_outputs.append(x.unsqueeze(1))
        
        return {
            "instances": decoder_outputs[0],  # ! Temporary fix, need to change this for multi shot
            "query_helpers": {
                "supp_feat_list": supp_feat_list,
                "final_supp_list": final_supp_list,
                "mask_list": mask_list
            }
        }
        
    def query_forward(self, image, prev_output, helpers):
        """
        Args:
            - prev_output: logits
        """
        # TODO: Merge previous output with the current image
        prev_output = torch.sigmoid(prev_output)
        x = self.query_input(torch.cat((image, prev_output), dim=1))

        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
        
        # Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0['pre_maxpool'](x)
            query_feat_0 = self.layer0['maxpool'](query_feat_0)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)  
            query_feat_4 = self.layer4(query_feat_3)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)
        
        # Merging with support features
        corr_query_mask_list = []
        supp_feat_list = helpers["supp_feat_list"]
        final_supp_list = helpers["final_supp_list"]
        mask_list = helpers["mask_list"]
        
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = mask_list[i]
            tmp_mask = tmp_mask[:, :, 0]
            tmp_mask = F.interpolate(tmp_mask, size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True) 

            tmp_supp = s               
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1) 
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1) 
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
            similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)   
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)  
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)     
        corr_query_mask = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)  

        supp_feat = supp_feat_list[0]
        if self.shot > 1:
            for i in range(1, len(supp_feat_list)):
                supp_feat += supp_feat_list[i]
            supp_feat /= len(supp_feat_list)

        out_list = []
        pyramid_feat_list = []

        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat)
            supp_feat_bin = supp_feat.expand(-1, -1, bin, bin)
            corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear', align_corners=True)
            merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, corr_mask_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx-1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx-1](rec_feat_bin) + merge_feat_bin  

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin   
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)
                 
        query_feat = torch.cat(pyramid_feat_list, 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat           
        out = self.cls(query_feat)
        
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        
        return { "masks": out.max(1)[1].unsqueeze(1) }
    
        # TODO: Auxilary loss later
        if self.training:
            y = helpers["q_gt"]
            main_loss = self.criterion(out, y.long())
            aux_loss = torch.zeros_like(main_loss).cuda()    

            for idx_k in range(len(out_list)):    
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss = aux_loss + self.criterion(inner_out, y.long())   
            aux_loss = aux_loss / len(out_list)
            return {
                "masks": out.max(1)[1],
                "losses": [main_loss, aux_loss]
            }
        else:
            return { "masks": out }

if __name__ == "__main__":
    import torch
    from isegm.model.ifss_pfenet_model import PFENetModel
    
    model = PFENetModel()
    B = 2
    H = 473
    W = 473
    query_image = torch.randn(B, 3, H, W)
    prev_query_mask = torch.randint(0, 2, (B, 1, H, W)).float()
    query_mask = torch.randint(0, 2, (B, 1, H, W)).float()
    
    support_image = torch.randn(B, 3, H, W)
    support_mask = torch.randint(0, 2, (B, 1, H, W)).float()
    
    support_output = model.support_forward(support_image, s_gt=support_mask)
    helpers = support_output["query_helpers"]
    helpers["q_gt"] = query_mask
    query_output = model.query_forward(query_image, prev_query_mask, helpers)
    
    print("Support Output:", support_output['instances'][0].shape)
    print("Query Output:", query_output['masks'].shape)