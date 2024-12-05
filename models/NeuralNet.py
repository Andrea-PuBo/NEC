import numpy as np
import matplotlib.pyplot as plt


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
        self.h = [np.zeros(n) for n in layers]  # Fields
        self.xi = [np.zeros(n) for n in layers]  # Activations
        #self.w = [None] + [np.random.randn(layers[i], layers[i - 1]) for i in range(1, self.L)]
        #self.theta = [None] + [np.random.randn(layers[i]) for i in range(1, self.L)]
        self.w = [None] + [np.random.randn(layers[i], layers[i-1]) * 0.01 for i in range(1, self.L)]  # Weights
        self.theta = [np.zeros(n) for n in layers]  # Thresholds
        self.delta = [np.zeros(n) for n in layers]  # Propagated errors
        self.d_w = [None] + [np.zeros_like(w) for w in self.w[1:]]  # Weight updates
        self.d_theta = [np.zeros(n) for n in layers]  # Threshold updates
        self.d_w_prev = [None] + [np.zeros_like(w) for w in self.w[1:]]  # Previous weight updates
        self.d_theta_prev = [np.zeros(n) for n in layers]  # Previous threshold updates

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

    def activation_derivative(self, h):
        # Compute the derivative of the activation function
        if self.fact == 'sigmoid':
            act = 1 / (1 + np.exp(-h))
            return act * (1 - act)
        elif self.fact == 'relu':
            return np.where(h > 0, 1, 0)
        elif self.fact == 'tanh':
            return 1 - np.tanh(h) ** 2
        elif self.fact == 'linear':
            return np.ones_like(h)

    def forward(self, X):
      # Compute forward propagation

      self.xi[0] = X  # Input layer activations
      for l in range(1, self.L):
        self.h[l] = np.dot(self.w[l], self.xi[l - 1]) - self.theta[l]
        self.xi[l] = self.activation(self.h[l])
      return self.xi[-1]  # Return output layer activations

    def backward(self, y_true):
      # Compute backward propagation

      # Compute delta for the output layer
      self.delta[-1] = (self.xi[-1] - y_true) * self.activation_derivative(self.h[-1])
      # Propagate errors backward
      for l in range(self.L - 2, 0, -1):
        self.delta[l] = np.dot(self.w[l + 1].T, self.delta[l + 1]) * self.activation_derivative(self.h[l])

    def update_weights_thresholds(self):
        # Update weights and thresholds using momentum
        for l in range(1, self.L):
            self.d_w[l] = -self.lr * np.outer(self.delta[l], self.xi[l - 1]) + self.momentum * self.d_w_prev[l]
            self.w[l] += self.d_w[l]
            self.d_w_prev[l] = self.d_w[l]

            self.d_theta[l] = self.lr * self.delta[l] + self.momentum * self.d_theta_prev[l]
            self.theta[l] += self.d_theta[l]
            self.d_theta_prev[l] = self.d_theta[l]

    def fit(self, X, y):
        # Train the neural network

        # Split data into training and validation sets
        n_train = int((1 - self.val_split) * len(X))
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        for epoch in range(self.epochs):
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_train, y_train = X_train[indices], y_train[indices]

            # Train on each sample
            for i in range(len(X_train)):
                self.forward(X_train[i])
                self.backward(y_train[i])
                self.update_weights_thresholds()

            # Compute losses
            train_loss = np.mean((self.predict(X_train) - y_train) ** 2)
            val_loss = np.mean((self.predict(X_val) - y_val) ** 2)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

        print("Returning from fit...")
        print("Train Losses:", self.train_losses)
        print("Validation Losses:", self.val_losses)

        # Return the recorded losses
        return self.train_losses, self.val_losses

    def predict(self, X):
        # Generate predictions
        predictions = []
        for sample in X:
            predictions.append(self.forward(sample))
        return np.array(predictions)

    def plot_errors(self):
        # Plot training and validation losses
        epochs = np.arange(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, label="Training error", marker='o')
        plt.plot(epochs, self.val_losses, label="Test error", marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error (MSE)") # Loss
        plt.legend()
        plt.title("Training and test error over epochs")
        plt.grid()
        plt.show()




