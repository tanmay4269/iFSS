from torch import nn
import segmentation_models_pytorch as smp


class SegmentationModel(nn.Module):
    def __init__(self, args):
        super(SegmentationModel, self).__init__()

        self.model = smp.DeepLabV3Plus(encoder_name="resnet34", in_channels=3, classes=2)


    def forward(self, images, targets=None):
        logits = self.model(images)

        if targets is None:
            return logits
        
        loss = nn.CrossEntropyLoss()(logits, targets)
        return logits, loss