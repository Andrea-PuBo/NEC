import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class NeuralNet:
    def __init__(self, layers, epochs, learning_rate, momentum, fact, val_split):
        self.L = len(layers)  # Number of layers
        self.n = layers       # Number of units in each layer
        self.epochs = epochs  # Number of training epochs
        self.lr = learning_rate  # Learning rate
        self.momentum = momentum  # Momentum term
        self.fact = fact  # Activation function
        self.val_split = val_split  # Validation set percentage

        # Initialize activations, weights, thresholds, and other variables
        self.h = [None] + [np.zeros(n) for n in layers[1:]]  # Fields
        self.xi = [None] + [np.zeros(n) for n in layers[1:]]  # Activations
        self.w = [None] + [np.random.randn(layers[i], layers[i - 1]) for i in range(1, self.L)]  # Weights
        self.theta = [None] + [np.random.randn(layers[i]) for i in range(1, self.L)]  # Thresholds
        self.delta = [None] + [np.zeros(n) for n in layers[1:]]  # Propagated errors
        self.d_w = [None] + [np.zeros_like(w) for w in self.w[1:]]  # Weight updates
        self.d_theta = [None] + [np.zeros(n) for n in layers[1:]]  # Threshold updates
        self.d_w_prev = [None] + [np.zeros_like(w) for w in self.w[1:]]  # Previous weight updates
        self.d_theta_prev = [None] + [np.zeros(n) for n in layers[1:]]  # Previous threshold updates

        # Loss tracking
        self.train_losses = []
        self.val_losses = []

    def activation(self, h):
        # Compute the activation function
        if self.fact == 'sigmoid':
            return 1 / (1 + np.exp(-h))
        elif self.fact == 'relu':
            return np.maximum(0, h)
        elif self.fact == 'tanh':
            return np.tanh(h)
        elif self.fact == 'linear':
            return h
        else:
            raise ValueError("Only valid functions: sigmoid, relu, tanh, linear")

    def activation_derivative(self, h):
        # Compute the derivative of the activation function
        if self.fact == 'sigmoid':
            act = 1 / (1 + np.exp(-h))
            return act * (1 - act)
        elif self.fact == 'relu':
            return np.where(h > 0, 1, 0)
        elif self.fact == 'tanh':
            return 1 - np.tanh(h)**2
        elif self.fact == 'linear':
            return np.ones_like(h)
        else:
            raise ValueError("Only valid functions: sigmoid, relu, tanh, linear")

    def forward(self, X):
        # Compute forward propagation

        self.xi[0] = X # Input layer activations
        for l in range(1, self.L):
            self.h[l] = np.dot(self.w[l], self.xi[l - 1]) - self.theta[l]
            self.xi[l] = self.activation(self.h[l])
        return self.xi[-1] # Return output layer activations

    def backward(self, y_true):
        # Compute backward propagation

        # Compute delta for the output layer
        self.delta[-1] = self.activation_derivative(self.h[-1]) * (self.xi[-1] - y_true)

        # Propagate errors backward. Compute deltas for hidden layers (excluding input layer)
        for l in range(self.L - 1, 1, -1):
            self.delta[l - 1] = self.activation_derivative(self.h[l - 1]) * np.dot(self.w[l].T, self.delta[l])

    def update_weights_thresholds(self):
        # Update weights and thresholds using momentum

        for l in range(1, self.L):
            self.d_w[l] = -self.lr * np.outer(self.delta[l], self.xi[l - 1])
            self.w[l] += self.d_w[l] + self.momentum * self.d_w_prev[l]
            self.d_w_prev[l] = self.d_w[l]

            self.d_theta[l] = self.lr * self.delta[l]
            self.theta[l] += self.d_theta[l] + self.momentum * self.d_theta_prev[l]
            self.d_theta_prev[l] = self.d_theta[l]

    def fit(self, X, y):
        # Train the neural network

        # Split data into training and validation sets
        n_train = int((1 - self.val_split) * len(X))
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        for epoch in range(self.epochs):
            for i in range(len(X_train)):
                self.forward(X_train[i])
                self.backward(y_train[i])
                self.update_weights_thresholds()

            # Compute losses
            train_loss = self.compute_error(X_train, y_train)
            val_loss = self.compute_error(X_val, y_val)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    def predict(self, X):
        # Generate predictions. Predict the output for a given input.
        predictions = []
        for x in X:
            self.forward(x)
            predictions.append(self.xi[-1])
        return np.array(predictions)

    def compute_error(self, X, y):
        # Compute the quadratic error
        total_error = 0
        num_samples = len(X)
        for x, output in zip(X, y):
            # Perform feed-forward for each pattern
            self.forward(x)
            # Compute quadratic error for this pattern
            total_error += np.sum((self.xi[-1] - output) ** 2)
        # Return the mean quadratic error across all patterns
        return total_error / num_samples

    def loss_epochs(self):
        # Return the evolution of the training loss and the validation loss
        return self.train_losses, self.val_losses

    def plot_errors(self):
        # Plot training and validation losses
        epochs = np.arange(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, label="Training Loss", marker='o')
        plt.plot(epochs, self.val_losses, label="Validation Loss", marker='o')
        plt.title("Training and test error over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.legend()
        plt.grid()
        plt.show()



if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv('../utils/A1-synthetic.csv', decimal=".")

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    columns = df_scaled.shape[1]

    input_columns = df_scaled.columns[0: 9]  # Select the first 9 columns
    features = df_scaled[input_columns].values

    output_column = df_scaled.columns[9]  # Select the 10th column (index 9) as the target
    targets = df_scaled[output_column].values

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    # Initialize Neural Network
    layers = [9, 8, 4, 1]  # Example: Input layer, two hidden layers, and output layer
    nn = NeuralNet(
        layers=layers,
        epochs=100,
        learning_rate=0.1,
        momentum=0.8,
        fact="sigmoid",  # Change activation as needed
        val_split=0.2
    )

    # Train the Neural Network
    nn.fit(X_train, y_train)

    # Get the training and validation errors
    train_errors, val_errors = nn.loss_epochs()

    # Plot the training and validation errors
    nn.plot_errors()





