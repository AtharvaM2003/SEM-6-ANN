import numpy as np
import matplotlib.pyplot as plt

# Sigmoid
def sigmoid(x):
    ''' It returns 1/(1+exp(-x)). where the values lie between zero and one '''
    return 1 / (1 + np.exp(-x))

# TanH
def tanh(x):
    ''' It returns the value (1-exp(-2x))/(1+exp(-2x)) and the value returned will lie between -1 to 1.'''
    return np.tanh(x)

# ReLU
def relu(x):
    ''' It returns zero if the input is less than zero otherwise it returns the given input. '''
    return np.maximum(0, x)

# Softmax
def softmax(x):
    ''' Compute softmax values for each set of scores in x. '''
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.linspace(-10, 10)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot each activation function
axs[0, 0].plot(x, sigmoid(x))
axs[0, 0].set_title('Activation Function: Sigmoid')

axs[0, 1].plot(x, tanh(x))
axs[0, 1].set_title('Activation Function: Tanh')

axs[1, 0].plot(x, relu(x))
axs[1, 0].set_title('Activation Function: ReLU')

axs[1, 1].plot(x, softmax(x))
axs[1, 1].set_title('Activation Function: Softmax')

# Adjust layout
plt.tight_layout()
plt.show()
