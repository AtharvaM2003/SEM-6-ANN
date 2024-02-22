import numpy as np

def bam(x):
    return np.sign(np.dot(W, x))

def backpropagation(y_target, x):
    return np.dot(W.T, y_target - bam(x))

X = np.array([[1, 1, 1, -1], [-1, -1, 1, 1]])
Y = np.array([[1, -1], [-1, 1]])
W = np.dot(Y.T, X)

x_test = np.array([1, -1, -1, -1])
y_target = bam(x_test)

delta_W = np.outer(y_target - bam(x_test), x_test)
W += delta_W

print("Input x:", x_test)
print("Target Output y:", y_target)
print("Updated Weight Matrix W:\n", W)
