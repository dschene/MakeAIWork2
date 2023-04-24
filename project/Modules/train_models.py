from Models import Lidar_sweep
from Models import Sonar_sweep
from torch.utils.data import DataLoader
from torch import nn

from tqdm import tqdm
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle


#######################################################

lidar_samples = pd.read_table('../Data/lidar_samples', sep = ' ', header=None)
sonar_samples = pd.read_table('../Data/lidar_samples', sep = ' ', header=None)

#######################################################

X_sonar = torch.Tensor(sonar_samples.iloc[:, :-1].to_numpy())
Y_sonar = torch.Tensor(sonar_samples.iloc[:, -1:].to_numpy())

X_lidar = torch.Tensor(lidar_samples.iloc[:, :-1].to_numpy())
Y_lidar = torch.Tensor(lidar_samples.iloc[:, -1:].to_numpy())

lidar_loader = DataLoader(list(zip(X_lidar, Y_lidar)), shuffle=False, batch_size=10)
sonar_loader = DataLoader(list(zip(X_sonar, Y_sonar)), shuffle=True, batch_size=10)

#######################################################

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

lidar_model = Lidar_sweep(16, 100, 75, 50, 25, 1)
sonar_model = Sonar_sweep()

#######################################################

learning_rate = 0.00005

loss_function = nn.MSELoss()
gradientDescent = torch.optim.SGD(lidar_model.parameters(), lr=learning_rate)

epochs = 10
losses = []

lidar_model.train()

for i in tqdm(range(epochs)):
    
    epoch_loss = []
    
    for x_batch, y_batch in lidar_loader:
        
        gradientDescent.zero_grad()
        
        y_pred = lidar_model(x_batch)
        
        loss = loss_function(y_pred, y_batch)
    
        loss.backward()
        
        gradientDescent.step()
        
        epoch_loss.append(loss.item())
        
        
    losses.append(sum(epoch_loss) / len(lidar_loader))


#######################################################

pickle.dump(lidar_model, open('model_1.pkl', 'wb'))