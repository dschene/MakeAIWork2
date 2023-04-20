import pandas as pd
import torch
import numpy
from torch import nn

class lidar_sweep(nn.Module):

    def __init__(self, inputs_size, hidden1_size, hidden2_size, outputs_size):
            super().__init__()

        self.model = nn.Sequential(
                nn.Linear(inputs_size, hidden1_size),
                nn.Linear(hidden1_size, hidden2_size),
                nn.Linear(hidden2_size, outputs_size)

    def forward(self, x):
        
        return self.model(x)





    