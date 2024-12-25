import numpy as np

class PerceptronModel():

  # Initializes the Perceptron Model
  def __init__(self, num_features):
    self.weights = [0.0] * num_features
    self.bias = 0.0
    # num_features = The number of features (input_dimensions) for the datasets
    # weights = a list of zeros with a size equal to the number of features. During training these weights will be updated.
    # bias = Our threshold is = 0.


  # Trains the given perceptron model using the given dataset.
  def train(self, X, y, learning_rate, num_iterations):
    for iteration in range(num_iterations):
      for i in range(len(X)):
        x = X[i]
        prediction = self.predict(x)
        error = y[i] - prediction
        for j in range(len(self.weights)):
          self.weights[j] += learning_rate * error * x[j]
        self.bias += learning_rate * error
    # x = a list of input data, where each element is a feature vector.
    # y = a list of target labels.
    # learning_rate = a small positive value used to control the step size of weight updates.
    # num_iterations = the number of times the model will iterate over the dataset. Think of it as how many epochs we will train our model.

  # Predicts the output for a given input
  def predict(self, x):
    sum = self.bias
    for i in range(len(self.weights)):
      sum+= self.weights[i] * x[i]
    return self.step_function(sum)
  # x = a single input.

  # Acts as the activation function for the perceptron
  def step_function(self, sum):
    return 1 if sum > self.bias else 0
  # If our sum is > than our bias our output will be 1, otherwise 0

  # Evaluates the accuracy of our perceptron model.
  def accuracy(self, X, y):
    correct = 0
    size_of_X = len(X)
    for i in range(size_of_X):
      x = X[i]
      prediction = self.predict(x)
      if prediction == y[i]:
        correct+=1
    return correct / size_of_X
  # x = a list of input data.
  # y = a list of true labels corresponding to the input data.
