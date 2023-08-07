import torchvision.models as models
from biotorch.models.utils import create_torchvision_biomodel
import torch
from torch import nn

MODE = 'fa'
MODE_STRING = 'Feedback Alignment'

class ClassifierMLP(torch.nn.Module):
    """ Simple MLP model
    input_feature: number of input feature
    hidden_layer_size: a list of units in hidden layer
    class_num: number of classes
    mode: normal, greedy, greedyExtraverts, intel
    extravert_mult, extravert_bias: used when mode is greedyExtraverts
    """

    def __init__(self, pretrained, progress, num_classes):
        super(ClassifierMLP, self).__init__()
        self.layers = []
        input_size = 784
        for i, h in enumerate([2000,2000,2000,2000]):
            self.layers.append(nn.Linear(input_size, h))
            self.layers.append(nn.ReLU())
            input_size = h
        self.layers.append(nn.Linear(input_size, num_classes))
        self.deep = nn.Sequential(*self.layers)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.deep(x)
        return x


def mlp(pretrained: bool = False, progress: bool = True, num_classes: int = 1000, layer_config=None) :
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): Output dimension of the last linear layer
        layer_config (dict): Custom biologically plausible method layer configuration
    """
    print('Converting MLP to {} mode'.format(MODE_STRING))
    return create_torchvision_biomodel(ClassifierMLP, MODE, layer_config, pretrained, progress, num_classes)
