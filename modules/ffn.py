import torch.nn as nn
from modules.activations import ACT2FN

class FeedForwardNetwork(nn.Module):

    def __init__(self, d_model, d_ff, activation='gelu') -> None:
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = ACT2FN[activation]

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

def main():
    import torch
    ffn = FeedForwardNetwork()
    x = torch.randn(4,5,8)
    ffn(x)