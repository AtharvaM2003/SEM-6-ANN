import numpy as np

# Step Activation Function
step_function = lambda x: 1 if x >= 0 else 0

# Training Data
training_data = [
    {'input': [1, 1, 0, 0, 0, 0], 'label': 1},
    {'input': [1, 1, 0, 0, 0, 1], 'label': 0},
    {'input': [1, 1, 0, 0, 1, 0], 'label': 1},
    {'input': [1, 1, 0, 1, 1, 1], 'label': 0},
    {'input': [1, 1, 0, 1, 0, 0], 'label': 1},
    {'input': [1, 1, 0, 1, 0, 1], 'label': 0},
    {'input': [1, 1, 0, 1, 1, 0], 'label': 1},
    {'input': [1, 1, 0, 1, 1, 1], 'label': 0},
    {'input': [1, 1, 1, 0, 0, 0], 'label': 1},
    {'input': [1, 1, 1, 0, 0, 1], 'label': 0},
]

# Initial Weights
weights = np.array([0, 0, 0, 0, 0, 1])

# Training the Perceptron
for data in training_data:
    input_data = np.array(data['input'])
    label = data['label']
    output = step_function(np.dot(input_data, weights))
    error = label - output
    weights += input_data * error

# Test the Perceptron for all Training Inputs
for data in training_data:
    input_to_test = data['input']
    output = "even" if step_function(np.dot(input_to_test, weights)) == 1 else "odd"
    print(f"The perceptron predicts {input_to_test} as {output}")
