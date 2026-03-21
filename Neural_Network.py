from copy import deepcopy

import numpy as np

from Perceptron import Perceptron

def loss(output_data:np.ndarray, expected_output:np.ndarray, derive=False):
    """
    Calculates the information loss for a given output data

    Args:
        output_data (np.ndarray): the actual output data from a neural network
        expected_output (np.ndarray): the expected output data
        derive (bool, optional): whether to calculate the derivative or not

    Returns:
        The squared sum of the difference between the output data and the expected output
    """
    if derive:
        return np.sum(2 * np.subtract(output_data, expected_output))

    return np.sum(np.square(np.subtract(output_data, expected_output)))

def softmax(data:np.ndarray):
    """
    A final activation step for the last output in a multi-layer neural network. It converts the outputs into probability distribution

    Args:
        data (np.ndarray): the final output of the neural network

    Returns:
        probability distribution of the output
    """
    e_x = np.exp(data - np.max(data))
    return e_x / np.sum(e_x)

def activation_function(input_data:np.ndarray, derive=False):
    """
    Adds non-linearity to the hypothesis in `linear_function()`. In this function, I implemented Rectified Linear Unit (ReLU)
    Args:
        input_data (np.ndarray): the input data
        derive(bool, optional): whether to calculate the derivative or not
    Returns:
        float: linear estimation
    """

    if derive:
        return np.array(input_data > 0)

    return np.array(np.maximum(0, input_data))

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
    weights_mat_list: list
    bias_list: list
    bet_outputs: list # list of linear outputs between the layers

    def __init__(self, input_size:int, output_size:int, hidden_layer_num:int, neurons_in_layers:list):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_num = hidden_layer_num
        self.neurons_in_layers = neurons_in_layers
        self.weights_mat_list = []
        self.bias_list = []
        self.bet_outputs = []

    def generate_weights(self, rows, columns):
        weights_mat = np.random.randn(rows, columns)
        self.weights_mat_list.append(weights_mat)
        return weights_mat

    def generate_bias(self, rows):
        bias_mat = np.random.randn(rows)
        self.bias_list.append(bias_mat)
        return bias_mat

    def train(self, input_data:np.ndarray, expected_output:np.ndarray):

        final = self.forward_pass(input_data)
        final = self.back_propagation(input_data, final, expected_output, 0.5)

        return final


    def forward_pass(self, input_data:np.ndarray):
        """
        Applies forward passing of input data and calculates the information loss for a given output data

        Args:
            input_data (np.ndarray): the input data

        Returns:
            final (np.ndarray): the output data
        """
        # initiate the start
        self.generate_weights(self.input_size, self.neurons_in_layers[0])
        self.generate_bias(self.neurons_in_layers[0])
        z_1 = input_data @ self.weights_mat_list[0] + self.bias_list[0]
        self.bet_outputs.append(z_1)
        output = activation_function(self.bet_outputs[-1])

        # hidden layers
        for i in range(self.hidden_layer_num - 1):
            self.generate_weights(self.neurons_in_layers[i], self.neurons_in_layers[i + 1])
            self.generate_bias(self.neurons_in_layers[i + 1])
            z_i = output @ self.weights_mat_list[-1] + self.bias_list[-1]
            self.bet_outputs.append(z_i)
            output = activation_function(self.bet_outputs[-1])

        # final step
        self.generate_weights(self.neurons_in_layers[-1], self.output_size)
        self.generate_bias(self.output_size)
        z_n = output @ self.weights_mat_list[-1] + self.bias_list[-1]
        self.bet_outputs.append(z_n)
        output = activation_function(self.bet_outputs[-1])

        # apply softmax to the final output of the neural network
        final = softmax(output)

        return final

    def back_propagation(self, input_data:np.ndarray, expected_output:np.ndarray, actual_output:np.ndarray, learning_rate:float):
        old_weights_mat_list = deepcopy(self.weights_mat_list)
        bet_outputs_deltas = []

        z_2_error = loss(expected_output, actual_output)
        z_2_delta = z_2_error * activation_function(actual_output, derive=True)
        bet_outputs_deltas.append(z_2_delta)
        self.weights_mat_list[-1] -= self.bet_outputs[-1].T.dot(z_2_delta) * learning_rate

        for i in range(1, self.hidden_layer_num + 1):
            z_i_error = bet_outputs_deltas[-1].dot(self.weights_mat_list[-i].T)
            z_i_delta = z_i_error * activation_function(self.bet_outputs[-(i+1)], derive=True)
            bet_outputs_deltas.append(z_i_delta)
            self.weights_mat_list[-(i+1)] -= self.bet_outputs[-(i+1)].T.dot(z_i_delta) * learning_rate

        # z_1_error = z_2_delta.dot(updated_weights_mat_list[-1].T)
        # z_1_delta = z_1_error * activation_function(self.bet_outputs[-2], derive=True)
        # updated_weights_mat_list[-2] -= self.bet_outputs[-2].T.dot(z_1_delta) * learning_rate

        print(self.weights_mat_list)
        print(old_weights_mat_list)



if __name__ == "__main__":
    print("Hello World")




