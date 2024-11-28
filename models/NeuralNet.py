import numpy as np


class NeuralNet:
    def __init__(self, layers, epochs, learning_rate, momentum, activation='sigmoid', val_split=0.2):
        self.L = len(layers)  # Number of layers
        self.n = layers       # Number of units in each layer
        self.epochs = epochs  # Number of training epochs
        self.lr = learning_rate  # Learning rate
        self.momentum = momentum  # Momentum term
        self.activation_name = activation  # Activation function
        self.val_split = val_split  # Validation set percentage

        # Initialize activations, weights, thresholds, and other variables
        self.h = [np.zeros(n) for n in layers]  # Fields
        self.xi = [np.zeros(n) for n in layers]  # Activations
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
      if self.activation_name == 'sigmoid':
        return 1 / (1 + np.exp(-h))
      elif self.activation_name == 'relu':
        return np.maximum(0, h)
      elif self.activation_name == 'tanh':
        return np.tanh(h)
      elif self.activation_name == 'linear':
        return h


    def activation_derivative(self, h):
      if self.activation_name == 'sigmoid':
        return self.activation(h) * (1 - self.activation(h))
      elif self.activation_name == 'relu':
        return (h > 0).astype(float)
      elif self.activation_name == 'tanh':
        return 1 - np.tanh(h) ** 2
      elif self.activation_name == 'linear':
        return np.ones_like(h)


    def forward(self, X):
      self.xi[0] = X  # Input layer activations
      for l in range(1, self.L):
        self.h[l] = np.dot(self.w[l], self.xi[l - 1]) - self.theta[l]
        self.xi[l] = self.activation(self.h[l])
      return self.xi[-1]  # Return output layer activations


    def backward(self, y_true):
      # Compute delta for the output layer
      self.delta[-1] = (self.xi[-1] - y_true) * self.activation_derivative(self.h[-1])

      # Propagate errors backward
      for l in range(self.L - 2, 0, -1):
        self.delta[l] = np.dot(self.w[l + 1].T, self.delta[l + 1]) * self.activation_derivative(self.h[l])

      # Update weights and thresholds
      for l in range(1, self.L):
        self.d_w[l] = -self.lr * np.outer(self.delta[l], self.xi[l - 1]) + self.momentum * self.d_w_prev[l]
        self.d_theta[l] = self.lr * self.delta[l] + self.momentum * self.d_theta_prev[l]

        self.w[l] += self.d_w[l]
        self.theta[l] += self.d_theta[l]

        self.d_w_prev[l] = self.d_w[l]
        self.d_theta_prev[l] = self.d_theta[l]


    def fit(self, X, y):
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

        # Compute losses
        train_loss = np.mean((self.predict(X_train) - y_train) ** 2)
        val_loss = np.mean((self.predict(X_val) - y_val) ** 2)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    def predict(self, X):
      predictions = []
      for sample in X:
        predictions.append(self.forward(sample))
      return np.array(predictions)


    def loss_epochs(self):
      return np.array([self.train_losses, self.val_losses]).T


layers = [4, 9, 5, 1]
nn = NeuralNet(layers, epochs=100, learning_rate=0.01, momentum=0.9)

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")
