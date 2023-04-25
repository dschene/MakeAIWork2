from torch import nn


class Lidar_sweep(nn.Module):
    
    def __init__(self, inputs_size, hidden1_size, hidden2_size, hidden3_size, hidden4_size, outputs_size):
        super().__init__()

        self.model = nn.Sequential(
                nn.Linear(inputs_size, hidden1_size),
                nn.ReLU(),
                nn.Linear(hidden1_size, hidden2_size),
                nn.ReLU(),
                nn.Linear(hidden2_size, hidden3_size),
                nn.ReLU(),
                nn.Linear(hidden3_size, hidden4_size),
                nn.ReLU(),
                nn.Linear(hidden4_size, outputs_size),
                                    )
        
    def forward(self, x):
        
        return self.model(x)

class Sonar_sweep(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        
        super().__init__()