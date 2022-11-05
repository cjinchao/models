import torch
import torch.nn as nn
from modules.activations import ACT2FN

class Dense(nn.Module):

    def __init__(self, in_fea, out_fea, activation='gelu') -> None:
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_fea, out_fea)
        self.activation = ACT2FN(activation)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x