import torch
import torch.nn as nn
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, in_channel, classes, bias=True, pretrained=True, **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.seq = models.resnet18(pretrained=pretrained)
        self.seq.fc = nn.Linear(512, classes, bias)

    def forward(self, x):
        if self.in_channel == 1:
            x = torch.cat((x, x, x), dim=1)
        x = self.seq(x)
        return x

    def freeze(self):
        self.seq.requires_grad_(False)
        self.seq.fc.requires_grad_(True)


if __name__ == '__main__':
    model = ResNet18(3, 10)
    for name, param in model.named_parameters():
        print(f"{name}_{param.data.size()}")
