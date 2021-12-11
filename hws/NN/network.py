import numpy as np
from neuron import Neuron


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class BasicNet:

    def __init__(self, input_w, input_b, output_w, output_b, inputs, outputs):
        self.hidden_w = input_w
        self.hidden_b = input_b
        self.out_w = output_w
        self.out_b = output_b
        self.X = inputs
        self.y = outputs
        self.results = [np.empty(2), np.empty(1)]

    def forward(self):
        self.results[0] = sigmoid(np.dot(self.X, self.hidden_w) + np.array([self.hidden_b] * 4).reshape(4, 2))
        self.results[1] = sigmoid(np.dot(self.results[0], self.out_w) + np.array([self.out_b] * 4).reshape(4, 1))
        return self.results[1]

    def backward(self):
        d_outw = np.dot(self.results[0].T, (self.y - self.results[1]) * sigmoid_derivative(self.results[1]))
        d_outb = np.sum((self.y - self.results[1]) * sigmoid_derivative(self.results[1]))

        d_inter = np.dot((self.y - self.results[1]) * sigmoid_derivative(self.results[1]), self.out_w.T)

        d_hiddenw = np.dot(self.X.T, d_inter * sigmoid_derivative(self.results[0]))
        d_hiddenb = np.sum(d_inter * sigmoid_derivative(self.results[0]))

        self.out_w += d_outw
        self.out_b += d_outb

        self.hidden_w += d_hiddenw
        self.hidden_b += d_hiddenb

class Net:
    def __init__(self, *args : Neuron):
        self.neurons = args

    def forward(self, x, y):
        for neuron in self.neurons:
            x = neuron.forward_prop(x, y)
        return y

    def backward(self):
        interd = None
        for neuron in self.neurons[::-1]:
            interd = neuron.backward_prop(interd)

