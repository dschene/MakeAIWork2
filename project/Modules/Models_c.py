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
        
    def forward(self, X):
        
        return self.model(X)


class Sonar_sweep(nn.Module):

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden1_size)
        self.activ1 = nn.ReLU()
        self.l2 = nn.Linear(hidden1_size, hidden2_size)
        self.activ2 = nn.ReLU()
        self.l3 = nn.Linear(hidden2_size, output_size)

    def forward(self, X):
        X = self.l1(X)
        X = self.activ1(X)
        X = self.l2(X)
        X = self.activ2(X)
        X = self.l3(X)
        
        return X