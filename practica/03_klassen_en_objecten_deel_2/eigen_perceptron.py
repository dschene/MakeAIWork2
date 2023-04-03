import numpy as np

class Perceptron:

    def __init__(self, learning_rate = 0.01, n_iterations = 1000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)

    def predict(self, X):



    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)
    

    
    