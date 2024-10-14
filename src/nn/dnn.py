import numpy as np
import scipy
import scipy.io
import torch
from collections import OrderedDict


class DNN(torch.nn.Module):
    """DNN Class"""

    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh

        # Layers
        layer_list = list()
        for i in range(self.depth - 1):
            w_layer = torch.nn.Linear(layers[i], layers[i + 1], bias=True)
            torch.nn.init.xavier_normal_(w_layer.weight)
            layer_list.append(("layer_%d" % i, w_layer))
            layer_list.append(("activation_%d" % i, self.activation()))

        w_layer = torch.nn.Linear(layers[-2], layers[-1], bias=True)
        torch.nn.init.xavier_normal_(w_layer.weight)
        layer_list.append(("layer_%d" % (self.depth - 1), w_layer))
        layerDict = OrderedDict(layer_list)
        # Deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out
