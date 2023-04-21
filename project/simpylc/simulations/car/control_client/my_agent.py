import time as tm
import traceback as tb
import math as mt
import sys as ss
import os
import socket as sc
import pickle 
import torch
import socket_wrapper as sw
import parameters as pm

lidar_model_1 = pickle.load(open('simulations/car/control_client/models/model_1.pkl', 'rb'))
sonar_model_1 = None

ss.path +=  [os.path.abspath (relPath) for relPath in  ('..',)] 


class HardcodedClient:
    def __init__ (self):
        self.steeringAngle = 0

        with open (pm.sampleFileName, 'w') as self.sampleFile:
            with sc.socket (*sw.socketType) as self.clientSocket:
                self.clientSocket.connect (sw.address)
                self.socketWrapper = sw.SocketWrapper (self.clientSocket)
                self.halfApertureAngle = False

                while True:
                    self.input ()
                    self.sweep ()
                    self.output ()
                    self.logTraining ()
                    tm.sleep (0.02)

    def input (self):
        sensors = self.socketWrapper.recv ()

        if not self.halfApertureAngle:
            self.halfApertureAngle = sensors ['halfApertureAngle']
            self.sectorAngle = 2 * self.halfApertureAngle / pm.lidarInputDim
            self.halfMiddleApertureAngle = sensors ['halfMiddleApertureAngle']
            
        if 'lidarDistances' in sensors:
            self.lidarDistances = sensors ['lidarDistances']
            #set model to sonar/lidar choice 
            self.model = lidar_model_1
        else:
            self.sonarDistances = sensors ['sonarDistances']
            #set model to sonar/lidar choice 
            self.model = sonar_model_1

    def lidarSweep (self):
        sample = [pm.finity for entryIndex in range (pm.lidarInputDim + 1)]

        for lidarAngle in range (-self.halfApertureAngle, self.halfApertureAngle):
            sectorIndex = round (lidarAngle / self.sectorAngle)
            sample [sectorIndex] = min (sample [sectorIndex], self.lidarDistances [lidarAngle])

        lidar_data = sample[:16]

        input_tensor = torch.tensor(lidar_data)

        self.steeringAngle = self.model(input_tensor).item()
        self.targetVelocity = pm.getTargetVelocity (self.steeringAngle)

    def sonarSweep (self):
        sample = [pm.finity for entryIndex in range (pm.sonarInputDim + 1)]

        for entryIndex, sectorIndex in ((2, -1), (0, 0), (1, 1)):
            sample [entryIndex] = self.sonarDistances [sectorIndex]

        sample [-1] = self.steeringAngle
        

        sonar_data = sample[:-2]

        input_tensor = torch.tensor(sonar_data)

        self.steeringAngle = self.model(input_tensor).item()
        self.targetVelocity = pm.getTargetVelocity (self.steeringAngle)
        
    def sweep (self):
        if hasattr (self, 'lidarDistances'):
            self.lidarSweep ()
        else:
            self.sonarSweep ()

    def output (self):
        actuators = {
            'steeringAngle': self.steeringAngle,
            'targetVelocity': self.targetVelocity
        }

        self.socketWrapper.send (actuators)

    def logLidarTraining (self):
        sample = [pm.finity for entryIndex in range (pm.lidarInputDim + 1)]

        for lidarAngle in range (-self.halfApertureAngle, self.halfApertureAngle):
            sectorIndex = round (lidarAngle / self.sectorAngle)
            sample [sectorIndex] = min (sample [sectorIndex], self.lidarDistances [lidarAngle])

        sample [-1] = self.steeringAngle
        print (*sample, file = self.sampleFile)

    def logSonarTraining (self):
        sample = [pm.finity for entryIndex in range (pm.sonarInputDim + 1)]

        for entryIndex, sectorIndex in ((2, -1), (0, 0), (1, 1)):
            sample [entryIndex] = self.sonarDistances [sectorIndex]

        sample [-1] = self.steeringAngle
        print (*sample, file = self.sampleFile)

    def logTraining (self):
        if hasattr (self, 'lidarDistances'):
            self.logLidarTraining ()
        else:
            self.logSonarTraining ()

HardcodedClient ()