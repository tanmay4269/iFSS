import torch
import torch.nn as nn

from SimpleClick.isegm.model.modeling.models_vit import VisionTransformer
from SimpleClick.isegm.model.is_plainvit_model import SimpleFPN
from SimpleClick.isegm.model.modeling.swin_transformer import SwinTransfomerSegHead


class PlainViTModel(nn.Module):
    def __init__(
        self,
        backbone_params={},
        neck_params={},
        head_params={},
        random_split=False,
    ):
        super().__init__()
        self.random_split = random_split

        self.backbone = VisionTransformer(**backbone_params)
        self.neck = SimpleFPN(**neck_params)
        self.head = SwinTransfomerSegHead(**head_params)

    def forward(self, image):
        backbone_features = self.backbone.forward_backbone(
            image, shuffle=self.random_split
        )

        # Extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape
        grid_size = self.backbone.patch_embed.grid_size

        backbone_features = backbone_features.transpose(-1, -2).view(
            B, C, grid_size[0], grid_size[1]
        )
        multi_scale_features = self.neck(backbone_features)

        return self.head(multi_scale_features)


if __name__ == '__main__':
    backbone_params = dict(
        img_size=(224,224),
        patch_size=(16,16),
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4, 
        qkv_bias=True,
    )

    neck_params = dict(
        in_dim = 768,
        out_dims = [256, 256, 256, 256],
    )

    head_params = dict(
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        loss_decode=nn.CrossEntropyLoss(),
        align_corners=False,
    )
    
    model = PlainViTModel(
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
    )
    
    sample_input = torch.randn(2, 3, 224, 224)  # B, C, H, W
    output = model.forward(sample_input)
    print(output.shape)  # torch.Size([2, 2, 56, 56])
