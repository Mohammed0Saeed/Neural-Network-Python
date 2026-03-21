from Perceptron import *
from Neural_Network import *
import numpy as np

input_data = np.random.random(3)
expected_output = np.array([0.2, 0.8])

neural_network = NeuralNetwork(3, 2, 4, [4, 8, 6, 4])

output = neural_network.train(input_data, expected_output)
# print(output)
