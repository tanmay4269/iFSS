import torch
from torch import nn
import segmentation_models_pytorch as smp


class SegmentationModel(nn.Module):
    def __init__(self, args):
        super(SegmentationModel, self).__init__()

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
