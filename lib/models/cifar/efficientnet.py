import torch.nn as nn

from efficientnet_pytorch import EfficientNet


class CEfficientNet(nn.Module):
    def __init__(self, num_classes=100, pretrained=False, model_name="efficientnet-b0"):
        super(CEfficientNet, self).__init__()
        if pretrained:
            self.features = EfficientNet.from_pretrained(model_name)
            print("pretrained")
        else:
            self.features = EfficientNet.from_name(model_name)
        self.features._conv_stem.stride = (1, 1)
        fc_features = self.features._fc.in_features
        self.features._fc = nn.Linear(fc_features, 100)
        # self.model_name = model_name

    def forward(self, x):
        out = self.features(x)
        return out


# Usage
def efficientnetb0(**kwargs):
    model = CEfficientNet(
        num_classes=100, pretrained=False, model_name="efficientnet-b0"
    )
    return model


def efficientnetb1(**kwargs):
    model = CEfficientNet(
        num_classes=100, pretrained=False, model_name="efficientnet-b1"
    )
    return model


def efficientnetb2(**kwargs):
    model = CEfficientNet(
        num_classes=100, pretrained=False, model_name="efficientnet-b2"
    )
    return model


def efficientnetb3(**kwargs):
    model = CEfficientNet(
        num_classes=100, pretrained=False, model_name="efficientnet-b3"
    )
    return model


def efficientnetb4(**kwargs):
    model = CEfficientNet(
        num_classes=100, pretrained=False, model_name="efficientnet-b4"
    )
    return model


def efficientnetb5(**kwargs):
    model = CEfficientNet(
        num_classes=100, pretrained=False, model_name="efficientnet-b5"
    )
    return model
