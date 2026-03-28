from copy import deepcopy
import cupy as cp
import os


def loss(output_data: cp.ndarray, expected_output: cp.ndarray, derive=False):
    """
    Calculates the information loss for a given output data

    Args:
        output_data (cp.ndarray): the actual output data from a neural network
        expected_output (cp.ndarray): the expected output data
        derive (bool, optional): whether to calculate the derivative or not

    Returns:
        The squared sum of the difference between the output data and the expected output
    """
    if derive:
        return cp.atleast_1d(2 * cp.subtract(output_data, expected_output))

    return cp.sum(cp.square(cp.subtract(output_data, expected_output)))


def softmax(data: cp.ndarray):
    """
    A final activation step for the last output in a multi-layer neural network. It converts the outputs into probability distribution

    Args:
        data (cp.ndarray): the final output of the neural network

    Returns:
        probability distribution of the output
    """
    e_x = cp.exp(data - cp.max(data))
    return e_x / cp.sum(e_x)


def activation_function(input_data: cp.ndarray, derive=False):
    """
    Adds non-linearity to the hypothesis in `linear_function()`. In this function, I implemented Rectified Linear Unit (ReLU)
    Args:
        input_data (cp.ndarray): the input data
        derive(bool, optional): whether to calculate the derivative or not
    Returns:
        float: linear estimation
    """

    if derive:
        return cp.where(input_data > 0, 1, 0.01)

    return cp.where(input_data > 0, input_data, 0.01 * input_data)


class NeuralNetwork:
    """
    A simple implementation of a neural network with chosen number of hidden layers

    Args:
        input_size (int): the size of input data
        output_size (int): the size of output data
        hidden_layer_num (int): number of hidden layers (exclusive input and output)
        neurons_in_layers (list): the number of neurons in each layer ordered in a list, i.e. if `hidden_layer_num` = 3, then neurons_in_layers = [3, 5, 3]
        model_name (str): the name of the model, for the directory of the weights and biases
    """
    input_size: int
    output_size: int
    hidden_layer_num: int
    accuracy: float
    neurons_in_layers: list
    weights_mat_list: list
    bias_list: list
    bet_outputs: list  # list of linear outputs between the layers
    model_name: str

    def __init__(self, input_size: int, output_size: int, hidden_layer_num: int, neurons_in_layers: list,
                 model_name: str):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_num = hidden_layer_num
        self.neurons_in_layers = neurons_in_layers
        self.weights_mat_list = []
        self.bias_list = []
        self.bet_outputs = []
        self.accuracy = 0
        self.model_name = model_name
        # Check if the model already exists
        try:
            self.load()
            print("Weights and biases are loaded")
        # Create a new model and save it
        except:
            self.generate_bias()
            self.generate_weights()
            print("Weights and biases are generated")
            self.save()
            self.load()

    def generate_weights(self):
        """
        Generates random weights for the neural network
        """
        # initiate the start
        self.weights_mat_list.append(
            cp.random.randn(self.input_size, self.neurons_in_layers[0]) * cp.sqrt(2 / self.input_size))

        # hidden layers
        for i in range(self.hidden_layer_num - 1):
            self.weights_mat_list.append(
                cp.random.randn(self.neurons_in_layers[i], self.neurons_in_layers[i + 1]) * cp.sqrt(
                    2 / self.neurons_in_layers[i]))

        # final step
        self.weights_mat_list.append(
            cp.random.randn(self.neurons_in_layers[-1], self.output_size) * cp.sqrt(2 / self.neurons_in_layers[-1]))

    def generate_bias(self):
        """
        Generates random biases for the neural network
        """
        # initiate the start
        self.bias_list.append(cp.random.randn(self.neurons_in_layers[0]))

        # hidden layers
        for i in range(self.hidden_layer_num - 1):
            self.bias_list.append(cp.random.randn(self.neurons_in_layers[i + 1]))

        # final step
        self.bias_list.append(cp.random.randn(self.output_size))

    def save(self):
        """
        Saves the weights and biases of the neural network to a directory
        """
        os.makedirs(f"Models/{self.model_name}", exist_ok=True)
        for i, (weight, bias) in enumerate(zip(self.weights_mat_list, self.bias_list)):
            cp.save(f"Models/{self.model_name}/weight_{i}.npy", weight)
            cp.save(f"Models/{self.model_name}/bias_{i}.npy", bias)
        print(f"Model saved to Models/{self.model_name}")

    def load(self):
        """
        Loads the weights and biases of the neural network from a directory
        """
        self.weights_mat_list = []
        self.bias_list = []
        for i in range(len(self.neurons_in_layers) + 1):
            self.weights_mat_list.append(cp.load(f"Models/{self.model_name}/weight_{i}.npy"))
            self.bias_list.append(cp.load(f"Models/{self.model_name}/bias_{i}.npy"))
        print(f"Model loaded from Models/{self.model_name}")

    def train(self, input_dataset: cp.ndarray, expected_output_dataset: cp.ndarray, loop: int,
              batch_size: int = 32):
        """
        Trains the model on the input data by shuffling the data and choosing random sample for training
        :param input_dataset: input data for training
        :param expected_output_dataset: expected output for optimization
        :param loop: the number of iteration on the sample data
        :param batch_size: the size of the sample picked up from the input data
        """
        num_samples = input_dataset.shape[0]

        for i in range(loop):
            # Shuffle data every epoch
            indices = cp.random.permutation(num_samples)
            input_dataset = input_dataset[indices]
            expected_output_dataset = expected_output_dataset[indices]
            batch_loss = 0

            # Process in batches
            for start in range(0, num_samples, batch_size):
                batch_input = input_dataset[start:start + batch_size]
                batch_output = expected_output_dataset[start:start + batch_size]

                # accumulate gradients over batch
                for j in range(batch_input.shape[0]):
                    final = self.forward_pass(batch_input[j])
                    self.back_propagation(batch_input[j], batch_output[j], final, 0.001)
                    batch_loss += loss(final, batch_output[j])

            print(f"Epoch: {i + 1}, Loss: {float(batch_loss / num_samples):.4f}")

        # Save the model after training
        self.save()
        print("Weights and biases are saved successfully")

    def accuracy_measure(self, test_data: cp.ndarray, expected_output: cp.ndarray, tolerance: float):
        """
        Measures the accuracy of the neural network based on the test sample
        :param test_data: the data points for the test
        :param expected_output: expected output for each test point
        :param tolerance: the margin of accepted errors
        """
        prediction = cp.array([self.forward_pass(test_data[i]) for i in range(test_data.shape[0])])
        correct = cp.sum(cp.abs(prediction - expected_output) <= tolerance)
        self.accuracy = (correct / test_data.shape[0])

    def predict(self, data):
        """
        Predicts the output of input data
        :param data: the input of the neural network
        :return: the expected score along with the accuracy of the prediction
        """
        final_score = self.forward_pass(data)
        decision = cp.argmax(final_score)
        print(final_score)
        return f"Score: {decision}, Accuracy of Model: {self.accuracy}"

    def forward_pass(self, input_data: cp.ndarray):
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

    def back_propagation(self, input_data: cp.ndarray, expected_output: cp.ndarray, actual_output: cp.ndarray,
                         learning_rate: float):
        """
        Back propagates the loss from output back to input and then updates the weights and biases based on learning rate
        :param input_data: input data points
        :param expected_output: expected output for each data point
        :param actual_output:
        :param learning_rate:
        """
        bet_outputs_deltas = []
        # final layer
        output_delta = loss(actual_output, expected_output, derive=True) * activation_function(self.bet_outputs[-1],
                                                                                               derive=True)
        bet_outputs_deltas.append(output_delta)

        # Calculating the derivative for each hidden layer
        for i in range(1, self.hidden_layer_num + 1):
            layer_i_delta = (self.weights_mat_list[-i] @ bet_outputs_deltas[-1]) * activation_function(
                self.bet_outputs[-(i + 1)], derive=True)
            bet_outputs_deltas.append(layer_i_delta)

        # Calculating the weight and bias gradient
        for i in range(self.hidden_layer_num + 1):
            if i == self.hidden_layer_num:
                activated_output = input_data
            else:
                activated_output = activation_function(self.bet_outputs[-(i + 2)])

            weight_gradient = cp.outer(activated_output, bet_outputs_deltas[i])
            bias_gradient = bet_outputs_deltas[i]

            weight_gradient = cp.clip(weight_gradient, -1, 1)
            bias_gradient = cp.clip(bias_gradient, -1, 1)

            # Updating the weights
            self.weights_mat_list[-(i + 1)] -= learning_rate * weight_gradient
            self.bias_list[-(i + 1)] -= learning_rate * bias_gradient


if __name__ == "__main__":
    print("Hello World")