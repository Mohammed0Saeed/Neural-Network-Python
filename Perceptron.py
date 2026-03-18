import numpy as np

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
    bias: float

    def __init__(self, inputs:np.ndarray, weights:np.ndarray, bias:float):
        self._inputs = inputs
        self._weights = weights
        self._bias = bias

    def linear_function(self):
        """
        Returns the hypothesis of the input data given weights and bias

        Returns:
           np.float: linear estimation
        """
        return np.dot(self._inputs, self._weights) + self._bias


    def activation_function(self):
        """
        Adds non-linearity to the hypothesis in `linear_function()`. In this function, I implemented Rectified Linear Unit (ReLU)

        Returns:
            float: linear estimation
        """
        return max(0, self.linear_function())

    def __repr__(self):
        return f"<{self.activation_function()}>"

if __name__ == "__main__":
    print("Perceptron")