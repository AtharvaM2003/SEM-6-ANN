import numpy as np

class BAM:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.zeros((input_size, output_size), dtype=int)

    def train(self, input_vectors, output_vectors):
        for i in range(len(input_vectors)):
            input_vector = np.array(input_vectors[i]).reshape((self.input_size, 1))
            output_vector = np.array(output_vectors[i]).reshape((self.output_size, 1))
            self.weights += input_vector.dot(output_vector.T)

    def recall(self, input_vector):
        input_vector = np.array(input_vector).reshape((self.input_size, 1))
        output_vector = self.weights.T.dot(input_vector)
        output_vector[output_vector >= 0] = 1
        output_vector[output_vector < 0] = -1
        return output_vector.flatten()

# Example usage with two pairs of vectors
input_vectors = [
    [1, -1, 1],
    [-1, 1, -1]
]

output_vectors = [
    [1, -1],
    [-1, 1]
]

bam = BAM(input_size=len(input_vectors[0]), output_size=len(output_vectors[0]))

# Training the BAM
bam.train(input_vectors, output_vectors)

# Test recall
test_input = [1, -1, 1]
recalled_output = bam.recall(test_input)

print(f"Test Input: {test_input}")
print(f"Recalled Output: {recalled_output}")
