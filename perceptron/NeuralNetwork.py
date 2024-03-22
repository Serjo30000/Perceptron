import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, rangeWeights):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.weights = []
        layer_input_size = input_size
        for layer_output_size in hidden_sizes + [output_size]:
            self.weights.append(np.random.uniform(-rangeWeights, rangeWeights, (layer_input_size, layer_output_size)))
            layer_input_size = layer_output_size

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, inputs):
        layer_output = inputs
        for weight in self.weights:
            layer_output = self.sigmoid(np.dot(layer_output, weight))
        return layer_output

    def backward(self, inputs, targets, learning_rate):
        layer_outputs = [inputs]
        layer_input = inputs
        for weight in self.weights:
            layer_input = self.sigmoid(np.dot(layer_input, weight))
            layer_outputs.append(layer_input)

        output_error = targets - layer_outputs[-1]
        output_delta = output_error * self.sigmoid_derivative(layer_outputs[-1])

        for i in range(len(self.weights) - 1, 0, -1):
            hidden_error = output_delta.dot(self.weights[i].T)
            hidden_delta = hidden_error * self.sigmoid_derivative(layer_outputs[i])
            self.weights[i] += np.outer(layer_outputs[i], output_delta) * learning_rate
            output_delta = hidden_delta

        self.weights[0] += np.outer(inputs, output_delta) * learning_rate

    def train(self, inputs, targets, learning_rate, epochs):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                self.backward(inputs[i], targets[i], learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(targets - self.feedforward(inputs)))
                print(f'Epoch {epoch}: loss {loss}')

    def conversionToFloat(self, inputs):
        category_mapping = {"Big": 0, "Middle": 1, "Little": 2}

        for row in inputs:
            category_column_index = len(row) - 1
            category = row[category_column_index]
            if category in category_mapping:
                row[category_column_index] = category_mapping[category]
            else:
                row[category_column_index] = len(category_mapping)

        inputs = np.array(inputs, dtype=float)
        return inputs

    def normalize(self, inputs):
        min_vals1 = np.min(inputs, axis=0)
        max_vals1 = np.max(inputs, axis=0)
        normalized_data = (inputs - min_vals1) / (max_vals1 - min_vals1)
        return normalized_data
