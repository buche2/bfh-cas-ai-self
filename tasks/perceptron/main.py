import numpy as np
import torch
from sklearn.metrics import accuracy_score


import random
def y(x, m, b):
  return m*x*x + b
  #return m * x + b

nb = 100
ma = 2000
#100
X = np.linspace(0, 10, nb)
#40
y_above = [y(x, 10, 5) + abs(random.gauss(20,ma)) for x in X]
y_below = [y(x, 10, 5) - abs(random.gauss(20,ma)) for x in X]
import matplotlib.pyplot as plt
plt.scatter(X, y_below, c='b')
plt.scatter(X, y_above, c='g')
plt.plot(X, y(X, 10, 5),linestyle='solid', c='r', linewidth=3, label='decision boundary')
plt.legend()
plt.show()

#100
x_1 = np.linspace(0, 10, nb)

x_2 = np.array([y(elem, 10, 5) + abs(random.gauss(20,ma)) for elem in x_1])
class_ones = np.column_stack((x_1, x_2))

x_2 = np.array([y(elem, 10, 5) - abs(random.gauss(20,ma)) for elem in x_1])
class_zeros = np.column_stack((x_1, x_2))

training_inputs = np.vstack((class_ones, class_zeros))

print(training_inputs.shape)

labels = np.hstack((np.ones(nb), np.zeros(nb))).T
labels.shape

from perceptron import Perceptron

perceptron = Perceptron(2, epochs=100, learning_rate=0.01)
perceptron.train(training_inputs, labels)

print(perceptron.predict_batch(class_ones))

print(perceptron.predict_batch(class_zeros))

print(perceptron.weights[1:])

print(perceptron.weights)

print(perceptron.weights[0])