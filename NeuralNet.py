import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_grad(x):
    return x * (1 - x)


class NeuralNet:
    def __init__(self, layers):
        self.layers = layers
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        layers = self.layers
        parameters = {}
        for i in range(1, len(layers)):
            parameters['W' + str(i)] = np.random.rand(layers[i], layers[i - 1])
            parameters['b' + str(i)] = np.random.rand(layers[i], 1)
        return parameters

    def forward_propagation(self, X, parameters):
        A = X
        caches = [(None, X)]
        layers = self.layers
        for i in range(1, len(layers)):
            W = parameters['W' + str(i)]
            b = parameters['b' + str(i)]
            Z = np.dot(W, A) + b
            A = sigmoid(Z)
            cache = (Z, A)
            caches.append(cache)
        return caches

    def compute_derivatives(self, parameters, caches, Y):
        m = len(Y)
        derivatives = {}
        Z, A = caches[-1]
        dA = -Y / A + (1 - Y) / (1 - A)
        layers = self.layers
        for i in range(len(layers) - 1, 0, -1):
            Z, A = caches[i]
            dZ = dA * sigmoid_grad(A)
            Z_prev, A_prev = caches[i - 1]
            derivatives['dW' + str(i)] = np.dot(dZ, A_prev.T) / m
            derivatives['db' + str(i)] = np.sum(dZ, axis=1, keepdims=True) / m
            W = parameters['W' + str(i)]
            dA = np.dot(W.T, dZ)
        return derivatives

    def back_propagation(self, parameters, derivatives, learning_rate=0.01):
        layers = self.layers
        for i in range(1, len(layers)):
            parameters['W' + str(i)] -= learning_rate * derivatives['dW' + str(i)]
            parameters['b' + str(i)] -= learning_rate * derivatives['db' + str(i)]
        return parameters

    def train(self, X, Y, num_iter=1000, learning_rate=0.01):
        parameters = self.parameters
        for i in range(num_iter):
            caches = self.forward_propagation(X, parameters)
            derivatives = self.compute_derivatives(parameters, caches, Y)
            parameters = self.back_propagation(parameters, derivatives, learning_rate)

    def predict(self, X):
        caches = self.forward_propagation(X, self.parameters)
        return caches[-1][1]


neural_net = NeuralNet([2, 3, 4, 1])
X = np.array([[0., 0., 1., 1.], [0., 1., 0., 1.]])
Y = np.array([[1., 0., 0., 1.]])
neural_net.train(X, Y, 1000, 0.5)

print(neural_net.predict(np.array([[0.], [1.]])))
