import imp
from operator import imod


import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()