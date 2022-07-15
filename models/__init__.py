from .resnet import ResNet18
from .vgg import vgg11_bn


def get_model(model_name, input_size, classes):
    if model_name == "ResNet18":
        return ResNet18(input_size[0], classes)
    elif model_name == "VGG11BN":
        return vgg11_bn(input_size[0], classes)

