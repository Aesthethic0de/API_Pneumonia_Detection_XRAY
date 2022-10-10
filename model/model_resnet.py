from torch import nn
from torchvision.models import resnet34

class Resnet_Pneumonia(nn.Module):

    def __init__(self, pretrained=True):
        super(Resnet_Pneumonia, self).__init__()
        self.backbone = resnet34(pretrained=pretrained)
        self.fc = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), 512)
        x = self.fc(x)
        return x