import numpy as np

# Define two pairs of vectors
x1 = np.array([1, 1, 1, -1])
y1 = np.array([1, -1])
x2 = np.array([-1, -1, 1, 1])
y2 = np.array([-1, 1])

# Compute weight matrix W
W = np.outer(y1, x1) + np.outer(y2, x2)

# Define BAM function
def bam(x):
    y = np.dot(W, x)
    y = np.where(y >= 0, 1, -1)
    return y

# Test BAM with input
x_test = np.array([1, -1, -1, -1])
y_test = bam(x_test)

# Print output
print("Input x: ", x_test)
print("Output y: ", y_test)
