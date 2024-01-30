import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels):
        for epoch in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

# Load Iris dataset
iris = load_iris()
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])

# Separate features and labels
X = data[['sepal length (cm)', 'sepal width (cm)']].values
y = (data['target'] == 0).astype(int)  # Binary classification (setosa or not)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create perceptron and train on the training data
perceptron = Perceptron(input_size=2)
perceptron.train(X_train, y_train)

# Plot the training data points
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='blue', label='Class 1')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='red', label='Not Setosa')

# Plot the decision boundary
x_values = np.linspace(min(X_train[:, 0]), max(X_train[:, 0]), 100)
y_values = -(perceptron.weights[0] + perceptron.weights[1] * x_values) / perceptron.weights[2]
plt.plot(x_values, y_values, label='Decision Boundary', linestyle='--', color='black')

plt.title('Perceptron Decision Boundary on Training Data')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.show()

# Plot the testing data points
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='blue', label='Class 0 (Setosa)')
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='red', label='Not Setosa')

# Plot the decision boundary
plt.plot(x_values, y_values, label='Decision Boundary', linestyle='--', color='black')

plt.title('Perceptron Decision Boundary on Testing Data')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.show()







