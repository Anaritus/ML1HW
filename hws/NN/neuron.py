import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class Neuron:

    def __init__(self, weights, bias):
        self.y = None
        self.output = None
        self.input = None
        self.weights = weights
        self.bias = bias

    def forward_prop(self, x_in, correct_y):
        self.y = correct_y
        self.input = x_in
        self.output = sigmoid(
            np.dot(self.input, self.weights) + np.array([self.bias] * self.input.shape[0]).reshape(self.input.shape[0],
                                                                                                   self.weights.shape[1]))

        return self.output

    def backward_prop(self, inter_d=None):
        if inter_d is None:  # that means that this is hidden layer
            dw = np.dot(self.input.T, (self.y - self.output) * sigmoid_derivative(self.output))
            db = np.sum((self.y - self.output) * sigmoid_derivative(self.output))
        else:  # that means that this is output layer
            dw = np.dot(self.input.T, inter_d * sigmoid_derivative(self.input))
            db = np.sum(inter_d * sigmoid_derivative(self.input))
        self.weights += dw
        self.bias += db

        if inter_d is None:
            return np.dot((self.y - self.output) * sigmoid_derivative(self.output), self.weights.T)
        else:
            return np.dot((self.y - self.output) * sigmoid_derivative(self.output), self.weights.T) * inter_d
