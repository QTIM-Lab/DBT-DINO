import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNet121(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.model.classifier = nn.Identity()

    def forward(self, x):
        return self.model(x)