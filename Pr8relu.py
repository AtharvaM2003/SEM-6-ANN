import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_layer_size)
        self.bias_hidden = np.zeros((1, self.hidden_layer_size))
        self.weights_hidden_output = np.random.randn(self.hidden_layer_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_propagation(self, X):
        # Hidden layer
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.relu(self.hidden_layer_input)

        # Output layer
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = self.softmax(self.output_layer_input)

        return self.output_layer_output

    def backward_propagation(self, X, y, output):
        # Output layer
        error_output = output - y
        gradient_output = error_output / len(X)

        # Hidden layer
        error_hidden = gradient_output.dot(self.weights_hidden_output.T)
        gradient_hidden = error_hidden * (self.hidden_layer_output > 0)

        # Update weights and biases
        self.weights_hidden_output -= self.learning_rate * self.hidden_layer_output.T.dot(gradient_output)
        self.bias_output -= self.learning_rate * np.sum(gradient_output, axis=0, keepdims=True)
        self.weights_input_hidden -= self.learning_rate * X.T.dot(gradient_hidden)
        self.bias_hidden -= self.learning_rate * np.sum(gradient_hidden, axis=0, keepdims=True)

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward_propagation(X)
            self.backward_propagation(X, y, output)

    def predict(self, X):
        output = self.forward_propagation(X)
        return np.argmax(output, axis=1)


# Example usage:

# Generate synthetic data for multi-class classification
np.random.seed(42)
X_train = np.random.rand(100, 10)  # 100 samples with 10 features
y_train = np.random.randint(0, 3, 100)  # 3 classes

# Convert labels to one-hot encoding
y_train_one_hot = np.eye(3)[y_train]

# Create and train the neural network
input_size = X_train.shape[1]
hidden_layer_size = 100
output_size = 3  # Number of classes
learning_rate = 0.01

nn = NeuralNetwork(input_size, hidden_layer_size, output_size, learning_rate)
nn.train(X_train, y_train_one_hot, epochs=1000)

# Make predictions
predictions = nn.predict(X_train)

# Print predictions
print("Predictions:", predictions)
