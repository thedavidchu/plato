"""
The ResNet-18 used by Fishing paper
"""
import torch.nn
import torch.nn as nn
import torchvision.models

from plato.config import Config


class Model(nn.Module):
    def __init__(self, num_classes=Config().trainer.num_classes):
        """Taken from breaching/cases/models/model_preparation.py"""
        pretrained = True
        super().__init__()
        self.body = torchvision.models.resnet18(pretrained=pretrained)
        self.fc = torch.nn.Linear(self.body.fc.in_features, num_classes)
        # NOTE(dchu): I'm not actually sure what this does...
        if pretrained:
            self.fc.weight.data = self.body.fc.weight[:num_classes]
            self.fc.bias.data = self.body.fc.bias[:num_classes]
        self.body.fc = self.fc
        assert not hasattr(self, "bias")

    def forward(self, x):
        out = self.body(x)
        return out

