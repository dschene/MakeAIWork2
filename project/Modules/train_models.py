from Models_c import Lidar_sweep, Sonar_sweep
from torch.utils.data import DataLoader
from torch import nn

from tqdm import tqdm
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle

#device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = 'cpu'

#######################################################
#reading in the data 

lidar_samples = pd.read_table('../Data/lidar_samples', sep = ' ', header=None)
sonar_samples = pd.read_table('../Data/sonar_samples', sep = ' ', header=None)

#######################################################
#creating tensors for X and Y of both sonar and lidar data

X_sonar = torch.Tensor(sonar_samples.iloc[:, :-1].to_numpy()).to(device)
Y_sonar = torch.Tensor(sonar_samples.iloc[:, -1:].to_numpy()).to(device)

X_lidar = torch.Tensor(lidar_samples.iloc[:, :-1].to_numpy()).to(device)
Y_lidar = torch.Tensor(lidar_samples.iloc[:, -1:].to_numpy()).to(device)

#Using DataLoader to iterate over the datasets and optionally shuffle + set batch size of training


#######################################################

lidar_loader = DataLoader(list(zip(X_lidar, Y_lidar)), shuffle=False, batch_size=10)
sonar_loader = DataLoader(list(zip(X_sonar, Y_sonar)), shuffle=True, batch_size=10)

#######################################################
#Creating the models with the layer parameters (layer counts are already defined in Models.py)

lidar_model = Lidar_sweep(16, 100, 75, 50, 25, 1).to(device)
sonar_model = Sonar_sweep(3, 10, 5, 1).to(device)

#######################################################
#Training lidar model

learning_rate_l = 0.00005

loss_function = nn.MSELoss()
gradientDescent = torch.optim.SGD(lidar_model.parameters(), lr=learning_rate_l)

epochs_l = 10
losses_l = []

lidar_model.train()

for i in tqdm(range(epochs_l)):
    epoch_loss = []

    for x_batch, y_batch in lidar_loader:
        gradientDescent.zero_grad()
        y_pred = lidar_model(x_batch)
        loss = loss_function(y_pred, y_batch)
        loss.backward()
        gradientDescent.step()
        epoch_loss.append(loss.item())
        
    losses_l.append(sum(epoch_loss) / len(lidar_loader))

#######################################################

pickle.dump(lidar_model, open('./models/lidar_model.pkl', 'wb'))

#######################################################
#Training sonar model

learing_rate_s = 0.000001

grad_desc = torch.optim.SGD(sonar_model.parameters(), lr=learing_rate_s)

epochs_s = 10
losses_s = []
sonar_model.train()

for i in tqdm(range(epochs_s)):
    ep_loss = []

    for x_batch, y_batch in sonar_loader:
        grad_desc.zero_grad()
        y_pred = sonar_model(x_batch)
        loss = loss_function(y_pred,y_batch)
        loss.backward()
        grad_desc.step()
        ep_loss.append(loss.item())
    
    losses_s.append(sum(ep_loss) / len(sonar_loader))
                    
pickle.dump(sonar_model, open('./models/sonar_model.pkl', 'wb'))