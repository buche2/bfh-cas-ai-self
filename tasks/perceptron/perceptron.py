import numpy as np

class Perceptron(object):

    def __init__(self, dim_inputs, epochs=100, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(dim_inputs + 1)  # plus 1 for bias

    def predict_batch(self, inputs):
        res_vector = np.dot(inputs, self.weights[1:]) + self.weights[0]
        activations = [1 if elem > 0 else 0 for elem in res_vector]
        return np.array(activations)

    def predict(self, inputs):
        res = np.dot(inputs, self.weights[1:]) + self.weights[0]
        # self.weights[0] is the bias
        if res > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs  # update weights
                self.weights[0] += self.learning_rate * (label - prediction)  # update bias
