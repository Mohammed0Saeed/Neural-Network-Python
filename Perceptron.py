import numpy as np
import random

class Perceptron:
    """
    A simple implementation of a neuron in a neural network

    Args:
        inputs(np.array): input data
        weights(np.array): weights added to input data
        bias(float): bias added to input data
    """
    inputs : np.ndarray
    weights : np.ndarray
    weights: np.ndarray
    bias: float

    # def __init__(self, inputs:np.ndarray, weights:np.ndarray, bias:float):
    #     self._inputs = inputs
    #     self._weights = weights
    #     self._bias = bias

    def __init__(self, inputs:np.ndarray):
        self._inputs = inputs
        self._weights = weights
        self._bias = random.random()

    @property
    def weights(self):
        return self._weights

    def update_weights(self, i, value):
        """
        Updates the weights given an index and a value
        :param i: the index of the weight (the corresponding neuron)
        :param value: the updated weight
        """
        self._weights[i] -= value

    @property
    def bias(self):
        return self._bias

    def update_bias(self, value):
        self._bias = value

    def linear_function(self):
        """
        Returns the hypothesis of the input data given weights and bias

        Returns:
           np.float: linear estimation
        """
        return np.dot(self._inputs, self._weights) + self._bias


    def activation_function(self, derive=False):
        """
        Adds non-linearity to the hypothesis in `linear_function()`. In this function, I implemented Rectified Linear Unit (ReLU)
        Args:
            derive(bool, optional): whether to calculate the derivative or not
        Returns:
            float: linear estimation
        """

        if derive:
            return int(self.linear_function() > 0)

        return max(0, self.linear_function())

    def backpropagation(self, d_loss:int, learning_rate:float):
        for i in range(len(self._inputs)):
            self.update_weights(i, learning_rate * d_loss * self.activation_function(derive=True) * self._inputs[i])

    def __repr__(self):
        return f"<{self.activation_function()}>"

if __name__ == "__main__":
    print("Perceptron")