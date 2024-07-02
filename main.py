# Import shits here

# What is a Neural Network
# input layer -> hidden layer -> output layer
# each layer contains a variety of Node
# each node has an activation function (e.g Sigmoid)

import numpy as np


class Node:
    def __init__(self, w_size=3, bias=0):
        self.weights = np.random.randn(w_size, 1)
        self.bias = bias

    def sigmoid_activate(self, x):
        return 1 / (1 + np.exp(-x))

    def compute(self, x):
        return self.sigmoid_activate(np.dot(self.weights.T, x) + self.bias)


class Layer:
    def __init__(self, num_neurons: int, neuron_size: int):
        self.num_neurons = num_neurons
        # Init node
        self.nodes = []
        for _ in range(num_neurons):
            self.nodes.append(Node(neuron_size))

    def compute(self, x):
        return np.array([el.compute(x) for el in self.nodes]).reshape(-1, 1)


class NeuralNetwork:
    def __init__(self, input_size: int, num_layers: list[int]):
        self.layers = []
        self.layers.append(Layer(num_layers[0], input_size))
        for index in range(1, len(num_layers)):
            self.layers.append(Layer(num_layers[index], num_layers[index - 1]))

    def compute(self, x):
        prev_layer_output = self.layers[0].compute(x)
        for layer in self.layers:
            prev_layer_output = layer.compute(prev_layer_output)
        return prev_layer_output


if __name__ == "__main__":
    nn = NeuralNetwork(2, [2, 1])
    # nn.visualize()
    print(nn.compute(np.array([[0], [1]])))
