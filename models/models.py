import torch
from torch import nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from .seg_models import QueryNet

class iSegModel(nn.Module):
    def __init__(self, args):
        super(iSegModel, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=8, kernel_size=1),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1),
        )

        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet34", in_channels=3, classes=2
        )

        # Freeze the encoder layers
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, images, click_masks, prev_preds, targets=None):
        input = torch.cat((click_masks, prev_preds.unsqueeze(1), images), dim=1).float()

        x = self.input_layer(input)

        features = self.model.encoder(x)
        decoder_output = self.model.decoder(*features)

        logits = self.model.segmentation_head(decoder_output)

        if targets is None:
            return logits

        loss = self.loss_fn(logits, targets)
        return logits, loss


class iFSSModel(nn.Module):
    def __init__(self, args):
        super(iFSSModel, self).__init__()

        # Support
        self.s_input_layer = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1),
        )

        self.s_net = smp.DeepLabV3Plus(
            encoder_name="resnet34", in_channels=3, classes=2
        )

        # Query
        self.q_input_layer = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1),
        )

        self.q_net = smp.DeepLabV3(
            encoder_name="resnet34", in_channels=3, classes=2
        )

        # Freeze the encoder layers
        for param in self.s_net.encoder.parameters():
            param.requires_grad = False
            
        for param in self.q_net.encoder.parameters():
            param.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss()
        self.mid_level_ft_idx = 4

    def forward(self, s_click_mask, s_prev_pred, x_s, y_s, q_prev_pred, x_q, y_q):
        s_input = torch.cat(
            (s_click_mask, s_prev_pred.unsqueeze(1), x_s), dim=1
        ).float()

        # Support Path
        x = self.s_input_layer(s_input)
        s_features = self.s_net.encoder(x)
        decoder_output = self.s_net.decoder(*s_features)
        s_logits = self.s_net.segmentation_head(decoder_output)

        s_pred = torch.max(s_logits, dim=1)[1]
        
        prototype = self.get_prototype(s_features, s_pred)

        # Query Path
        q_input = torch.cat((q_prev_pred.unsqueeze(1), x_q), dim=1)

        x = self.q_input_layer(q_input)
        q_features = self.q_net.encoder(x)

        q_ft = q_features[self.mid_level_ft_idx]
        _, _, H, W = q_ft.shape
        prototype = prototype.unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)

        q_ft = torch.cat((q_ft, prototype), dim=1)
        q_features = [q_ft]  # TODO: do this properly

        decoder_output = self.q_net.decoder(*q_features)
        q_logits = self.q_net.segmentation_head(decoder_output)

        # Loss
        s_loss = self.loss_fn(s_logits, y_s)
        q_loss = self.loss_fn(q_logits, y_q)
        loss = 0.5 * s_loss + 0.5 * q_loss


        return {
            'logits': [s_logits, q_logits],
            'losses': [s_loss, q_loss, loss],
        }

    def get_prototype(self, s_features, s_pred):
        features = s_features[self.mid_level_ft_idx]
        _, _, H, W = features.shape

        pred = F.interpolate(
            s_pred.unsqueeze(1).float(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        prototype = torch.sum(features * pred, dim=(2, 3))

        return prototype
