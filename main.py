from Perceptron import *
from Neural_Network import *
import numpy as np

input_data = np.random.random(10)
expected_output = np.array([0, 1, 2, 3, 4])

neural_network = NeuralNetwork(10, 5, 7, [120, 160, 180, 1000, 800, 100, 60])

output = neural_network.train(input_data, expected_output)
print(output)