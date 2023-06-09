#!/usr/bin/env python

from perceptron import Perceptron
import itertools
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

# Waarheidstabel invoerwaarden
possibleOutcomes = [0, 1]
xTrain = np.array(
    [element for element in itertools.product(possibleOutcomes, possibleOutcomes)]
)
# 2 inputs
#
# 0 0
# 0 1
# 1 0
# 1 1

logging.debug(f"xTrain : {xTrain}")

# Waarheidstabel output
# 0 0 -> 0
# 0 1 -> 0
# 1 0 -> 0
# 1 1 -> 1 

# yTrain = np.array([0, 0, 0, 1])

# # Maak een object andperceptron aan
# andPerceptron = Perceptron()
# # Train de perceptron met een AND functie
# andPerceptron.train(xTrain, yTrain, epochs=100, learningRate=0.1)

# # Test de perceptron
# testInput = np.array([1, 1])
# prediction = andPerceptron.predict(testInput)
# logging.info(f"Predicted y value : {prediction}")

# # Test de perceptron
# # testInput = np.array([1, 1])
# logging.debug(f"testInput : {testInput}")
# logging.info(f"Predicted y value : {prediction}")

# # prediction = andPerceptron.predict(testInput)
# logging.info(f"Predicted y value : {prediction}")

# OPDDRACHT
# Maak nu zelf het object orPerceptron
orPerceptron = Perceptron()

# Waarheidstabel output
# 0 0 -> 0
# 0 1 -> 1
# 1 0 -> 1
# 1 1 -> 1

yTrain = np.array([0, 1, 1, 1])
orPerceptron.train(xTrain, yTrain, epochs=100, learningRate=0.1)
testInput = np.array([0, 1])

prediction = orPerceptron.predict(testInput)
logging.debug(f"testInput : {testInput}")
logging.info(f"Predicted y value : {prediction}")

testInput = np.array([0, 1])
testInput = np.array([1, 0])
testInput = np.array([1, 1])



