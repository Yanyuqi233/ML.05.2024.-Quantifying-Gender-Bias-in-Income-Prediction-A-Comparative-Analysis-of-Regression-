import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def forward(self, inputs):
        # Perform forward pass
        hidden = np.dot(inputs, self.weights_input_hidden)
        hidden_activation = self.sigmoid(hidden)
        output = np.dot(hidden_activation, self.weights_hidden_output)
        return output

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))
    
    def train(self, X_train, y_train, learning_rate=0.01, epochs=100):
        mse_history = []
        mae_history = []

        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X_train)

            # Calculate MSE and MAE
            mse = mean_squared_error(y_train, output)
            mae = mean_absolute_error(y_train, output)

            # Store MSE and MAE
            mse_history.append(mse)
            mae_history.append(mae)

            # Backpropagation
            error = output - y_train
            delta_output = error
            delta_hidden = np.dot(delta_output, self.weights_hidden_output.T) * (output * (1 - output))

            # Update weights
            self.weights_hidden_output -= learning_rate * np.dot(hidden_activation.T, delta_output)
            self.weights_input_hidden -= learning_rate * np.dot(X_train.T, delta_hidden)

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, MSE: {mse}, MAE: {mae}")

        return mse_history, mae_history
    
    def get_weights(self):
        return self.weights_input_hidden, self.weights_hidden_output
    
class ComplexNN:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Initialize weights randomly
        self.weights = []
        layer_input_size = input_size
        for hidden_size in hidden_sizes:
            self.weights.append(np.random.randn(layer_input_size, hidden_size))
            layer_input_size = hidden_size
        self.weights.append(np.random.randn(layer_input_size, output_size))

    def forward(self, inputs):
        # Perform forward pass
        layer_input = inputs
        for weight in self.weights[:-1]:
            layer_output = np.dot(layer_input, weight)
            layer_input = self.sigmoid(layer_output)
        output = np.dot(layer_input, self.weights[-1])
        return output

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))
    
    def train(self, X_train, y_train, learning_rate=0.01, epochs=100):
        mse_history = []
        mae_history = []

        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X_train)

            # Calculate MSE and MAE
            mse = mean_squared_error(y_train, output)
            mae = mean_absolute_error(y_train, output)

            # Store MSE and MAE
            mse_history.append(mse)
            mae_history.append(mae)

            # Backpropagation
            error = output - y_train
            delta_output = error
            delta_hidden = []
            layer_input = X_train

            # Calculate deltas for hidden layers
            for i in range(len(self.weights) - 1, 0, -1):
                delta_hidden.append(np.dot(delta_output, self.weights[i].T) * (layer_input * (1 - layer_input)))
                delta_output = np.dot(delta_hidden[-1], self.weights[i].T)

            # Update weights
            for i in range(len(self.weights) - 1, 0, -1):
                self.weights[i] -= learning_rate * np.dot(self.sigmoid(np.dot(X_train, self.weights[i - 1])).T, delta_output)
                delta_output = delta_hidden.pop()

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, MSE: {mse}, MAE: {mae}")

        return mse_history, mae_history
    
    def get_weights(self):
        return self.weights
