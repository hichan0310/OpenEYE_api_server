import torchvision
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(1000, 10)

    def forward(self, images):
        outputs = self.model(images)
        outputs = self.dropout(outputs)
        outputs = self.classifier(outputs)
        return outputs
