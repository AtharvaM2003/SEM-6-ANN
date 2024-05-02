import numpy as np

class McCullochPittsNN:
    def __init__(self, num_inputs, weights):
        self.weights = np.array(weights)
        self.threshold = 0

    def activation_function(self, net_input):
        return int(net_input > self.threshold)

    def forward_pass(self, inputs):
        net_input = np.dot(inputs, self.weights)
        return self.activation_function(net_input)

def generate_ANDNOT():
    mp_nn = McCullochPittsNN(2, [-1, 1])
    truth_table = [(0, 0), (0, 1), (1, 0), (1, 1)]

    print("Truth Table for ANDNOT Function:")
    print("Input1\tInput2\tOutput")
    for inputs in truth_table:
        output = mp_nn.forward_pass(inputs)
        print(f"{inputs[0]}\t{inputs[1]}\t{output}")

def main():
    while True:
        print("\nMenu:")
        print("1. Generate ANDNOT Function")
        print("2. Exit")

        choice = input("Enter your choice: ")
        if choice == "1":
            generate_ANDNOT()
        elif choice == "2":
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()
