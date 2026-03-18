import numpy as np
import scipy as sp

from Perceptron import Perceptron

class NeuralNetwork:
    """
    A simple implementation of a neural network with chosen number of hidden layers

    Args:
        input_size (int): the size of input data
        output_size (int): the size of output data
        hidden_layer_num (int): number of hidden layers (exclusive input and output)
        neurons_in_layers (list): the number of neurons in each layer ordered in a list
    """
    input_size: int
    output_size: int
    hidden_layer_num: int
    neurons_in_layers: list

    def __init__(self, input_size:int, output_size:int, hidden_layer_num:int, neurons_in_layers:list):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_num = hidden_layer_num
        self.neurons_in_layers = neurons_in_layers

    def train(self, input_data:np.ndarray, expected_output:np.ndarray):
        """
        Applies forward passing of input data and calculates the information loss for a given output data

        Args:
            input_data (np.ndarray): the input data
            expected_output (np.ndarray): the expected output data

        Returns:
            output (np.ndarray): the output data
            loss (float): the information loss
        """
        # initiate the start
        layer_1 = np.array([Perceptron(input_data, np.random.random(self.input_size), 1) for _ in range(self.input_size)])
        output = np.array([layer_1[i].activation_function() for i in range(self.input_size)])

        # hidden layers
        for i in range(self.hidden_layer_num):
            prev_neurons_sum = self.neurons_in_layers[i - 1] if i > 0 else self.input_size
            neurons_num = self.neurons_in_layers[i]
            layer_i = np.array([Perceptron(output, np.random.random(prev_neurons_sum), 1) for _ in range(neurons_num)])
            output = np.array([layer_i[j].activation_function() for j in range(neurons_num)])

        # output
        layer_n = np.array([Perceptron(output, np.random.random(self.neurons_in_layers[-1]), 1) for _ in range(self.output_size)])
        output = np.array([layer_n[i].activation_function() for i in range(self.output_size)])

        result = sp.special.softmax(output)

        return f"output: {result}, loss: {self.loss(result, expected_output)}"

    def loss(self, output_data:np.ndarray, expected_output:np.ndarray):
        """
        Calculates the information loss for a given output data

        Args:
            output_data (np.ndarray): the actual output data from a neural network
            expected_output (np.ndarray): the expected output data

        Returns:
            The squared sum of the difference between the output data and the expected output
        """
        return np.sum(np.square(np.subtract(output_data, expected_output)))
