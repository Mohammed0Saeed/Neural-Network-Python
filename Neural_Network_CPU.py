from copy import deepcopy
import os

    """
    Calculates the information loss for a given output data

    Args:
        derive (bool, optional): whether to calculate the derivative or not

    Returns:
        The squared sum of the difference between the output data and the expected output
    """
    if derive:


    """
    A final activation step for the last output in a multi-layer neural network. It converts the outputs into probability distribution

    Args:

    Returns:
        probability distribution of the output
    """

    """
    Adds non-linearity to the hypothesis in `linear_function()`. In this function, I implemented Rectified Linear Unit (ReLU)
    Args:
        derive(bool, optional): whether to calculate the derivative or not
    Returns:
        float: linear estimation
    """

    if derive:


class NeuralNetwork:
    """
    A simple implementation of a neural network with chosen number of hidden layers

    Args:
        input_size (int): the size of input data
        output_size (int): the size of output data
        hidden_layer_num (int): number of hidden layers (exclusive input and output)
    """
    input_size: int
    output_size: int
    hidden_layer_num: int
    accuracy: float
    neurons_in_layers: list
    weights_mat_list: list
    bias_list: list
    bet_outputs: list  # list of linear outputs between the layers

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_num = hidden_layer_num
        self.neurons_in_layers = neurons_in_layers
        self.weights_mat_list = []
        self.bias_list = []
        self.bet_outputs = []
        self.accuracy = 0
            self.generate_bias()
            self.generate_weights()

        """
        Generates random weights for the neural network
        """
        # initiate the start
        self.weights_mat_list.append(

        # hidden layers
        for i in range(self.hidden_layer_num - 1):
            self.weights_mat_list.append(
                    2 / self.neurons_in_layers[i]))

        # final step
        self.weights_mat_list.append(

    def generate_bias(self):
        """
        Generates random biases for the neural network
        """
        # initiate the start

        # hidden layers
        for i in range(self.hidden_layer_num - 1):

        # final step

        """
        """
        for i in range(loop):


        """
        Measures the accuracy of the neural network based on the test sample
        :param test_data: the data points for the test
        :param expected_output: expected output for each test point
        :param tolerance: the margin of accepted errors
        """
        self.accuracy = (correct / test_data.shape[0])

    def predict(self, data):
        """
        Predicts the output of input data
        :param data: the input of the neural network
        :return: the expected score along with the accuracy of the prediction
        """

        """
        Applies forward passing of input data and calculates the information loss for a given output data
        :param input_data: input data points
        :return: output of the neural network
        """
        # reset pre-activation values between layers
        self.bet_outputs = []

        # initiate the start
        z_1 = input_data @ self.weights_mat_list[0] + self.bias_list[0]
        self.bet_outputs.append(z_1)
        output = activation_function(self.bet_outputs[-1])

        # hidden layers
        for i in range(1, self.hidden_layer_num):
            z_i = output @ self.weights_mat_list[i] + self.bias_list[i]
            self.bet_outputs.append(z_i)
            output = activation_function(self.bet_outputs[-1])

        # final step
        z_n = output @ self.weights_mat_list[-1] + self.bias_list[-1]
        self.bet_outputs.append(z_n)
        output = self.bet_outputs[-1]

        # apply softmax to the final output of the neural network
        score = output if self.output_size == 1 else softmax(output)
        return score

        """
        Back propagates the loss from output back to input and then updates the weights and biases based on learning rate
        :param input_data: input data points
        :param expected_output: expected output for each data point
        :param actual_output:
        :param learning_rate:
        """
        bet_outputs_deltas = []
        # final layer
        bet_outputs_deltas.append(output_delta)

        # Calculating the derivative for each hidden layer
        for i in range(1, self.hidden_layer_num + 1):
            bet_outputs_deltas.append(layer_i_delta)

        # Calculating the weight and bias gradient
        for i in range(self.hidden_layer_num + 1):
            if i == self.hidden_layer_num:
                activated_output = input_data
            else:
                activated_output = activation_function(self.bet_outputs[-(i + 2)])

            bias_gradient = bet_outputs_deltas[i]


            self.weights_mat_list[-(i + 1)] -= learning_rate * weight_gradient
            self.bias_list[-(i + 1)] -= learning_rate * bias_gradient


if __name__ == "__main__":
    print("Hello World")