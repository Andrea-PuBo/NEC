import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold

class NeuralNet:
    def __init__(self, layers, epochs, learning_rate, momentum, fact, val_split):
        self.L = len(layers)  # Number of layers
        self.n = layers       # Number of units in each layer
        self.epochs = epochs  # Number of training epochs
        self.lr = learning_rate  # Learning rate
        self.momentum = momentum  # Momentum term
        self.fact = fact  # Activation function
        self.val_split = val_split  # Validation set percentage

        np.random.seed(42)
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
        n_train = int((1 - self.val_split) * len(X))
        if self.val_split > 0:
            X_train, X_val = X[:n_train], X[n_train:]
            y_train, y_val = y[:n_train], y[n_train:]
        else:
            X_train, y_train = X, y  # Use the entire dataset as training data
            X_val, y_val = None, None  # No validation data

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
            train_loss = self.compute_error(X_train, y_train)
            if X_val is not None and y_val is not None:
                val_loss = self.compute_error(X_val, y_val)
            else:
                val_loss = None

            self.train_losses.append(train_loss)
            if val_loss is not None:
                self.val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss}", end="")
            if val_loss is not None:
                print(f", Val Loss: {val_loss}")
            else:
                print()

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
        for x, y in zip(X, y):
            # Perform feed-forward for each pattern
            self.forward(x)
            # Compute quadratic error for this pattern
            total_error += np.sum((self.xi[-1] - y) ** 2)
        # Return the mean quadratic error across all patterns
        return total_error / num_samples

    def loss_epochs(self):
        # Return the evolution of the training loss and the validation loss
        return self.train_losses, self.val_losses

    def cross_validate(self, X, y, k_folds=4):
        # Perform k-fold cross-validation on the neural network.

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        mse_list, mae_list, mape_list = [], [], []

        for fold, (train_index, val_index) in enumerate(kf.split(X)):
            print(f"Fold {fold + 1}/{k_folds}")
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Reset the model by reinitializing weights and thresholds
            self.__init__(self.n, self.epochs, self.lr, self.momentum, self.fact, self.val_split)

            # Train on training set
            self.fit(X_train, y_train)

            # Predict on validation set
            y_pred = self.predict(X_val)

            # Compute metrics for this fold
            # Mean Squared Error (MSE)
            total_error = 0
            for y_predicted, y_real in zip(y_pred, y_val):
                total_error += (y_real - y_predicted) ** 2
            mse = total_error / len(y_val)

            # Mean Absolute Error (MAE)
            total_error = 0
            for y_predicted, y_real in zip(y_pred, y_val):
                total_error += abs(y_real - y_predicted)
            mae = total_error / len(y_val)

            # Mean Absolute Percentage Error (MAPE)
            total_error = 0
            epsilon = 1e-7  # Avoid division by zero
            for y_predicted, y_real in zip(y_pred, y_val):
                if y_real != 0:
                    total_error += abs((y_real - y_predicted) / (y_real + epsilon))
            mape = (total_error / len(y_val)) * 100

            print(f"Fold {fold + 1} MSE: {mse}, MAE: {mae}, MAPE: {mape}")

            mse_list.append(mse)
            mae_list.append(mae)
            mape_list.append(mape)

        # Aggregate results
        results = {
            "MSE Mean": np.mean(mse_list),
            "MSE Std": np.std(mse_list),
            "MAE Mean": np.mean(mae_list),
            "MAE Std": np.std(mae_list),
            "MAPE Mean": np.mean(mape_list),
            "MAPE Std": np.std(mape_list)
        }

        return results

#Hyperparameter tuning with cross-validation
# if __name__ == "__main__":
#     # Load dataset
#     df = pd.read_csv('../data/preprocessed_data.csv', decimal=".")
#     output_column = 'price'
#     y = df[output_column].values
#     X = df.drop(columns=[output_column]).values
#
#     # Define hyperparameter combinations
#     hyperparameters = [
#         {"layers": [9, 8, 4, 1], "epochs": 100, "lr": 0.1, "momentum": 0.8, "activation": "sigmoid"},
#         {"layers": [9, 8, 4, 1], "epochs": 200, "lr": 0.1, "momentum": 0.2, "activation": "sigmoid"},
#         {"layers": [9, 32, 16, 8, 1], "epochs": 100, "lr": 0.01, "momentum": 0.8, "activation": "tanh"},
#         {"layers": [9, 32, 16, 8, 1], "epochs": 200, "lr": 0.01, "momentum": 0.2, "activation": "tanh"}
#     ]
#
#     results = []
#
#     for params in hyperparameters:
#         print(f"Testing configuration: {params}")
#
#         # Initialize Neural Network with current hyperparameters
#         nn = NeuralNet(
#             layers=params["layers"],
#             epochs=params["epochs"],
#             learning_rate=params["lr"],
#             momentum=params["momentum"],
#             fact=params["activation"],
#             val_split=0.0  # No validation split here, cross-validation handles it
#         )
#
#         # Perform cross-validation
#         cv_results = nn.cross_validate(X, y, k_folds=4)
#
#         # Store results
#         results.append({
#             "Layers": params["layers"],
#             "Epochs": params["epochs"],
#             "Learning Rate": params["lr"],
#             "Momentum": params["momentum"],
#             "Activation": params["activation"],
#             "MSE Mean": cv_results["MSE Mean"],
#             "MSE Std": cv_results["MSE Std"],
#             "MAE Mean": cv_results["MAE Mean"],
#             "MAE Std": cv_results["MAE Std"],
#             "MAPE Mean": cv_results["MAPE Mean"],
#             "MAPE Std": cv_results["MAPE Std"]
#         })
#
#         # Create DataFrame
#         results_df = pd.DataFrame(results)
#
#         # Save the results to a CSV file in the data folder
#         results_df.to_csv('../data/cross_validation_results.csv', index=False)
#
#         print("Results saved to ../data/cross_validation_results.csv")
#         print(results_df)

# Neural Network with best parameters
if __name__ == "__main__":
    #read the csv data
    X_train_pd = pd.read_csv('../data/X_train.csv')
    X_test_pd = pd.read_csv('../data/X_test.csv')
    y_train_pd = pd.read_csv('../data/y_train.csv')
    y_test_pd = pd.read_csv('../data/y_test.csv')

    # Convert X_train and X_test to NumPy arrays
    X_train = X_train_pd.to_numpy()
    X_test = X_test_pd.to_numpy()
    # Assuming y_train and y_test have a single target column
    y_train = y_train_pd.to_numpy().ravel()  # Use .ravel() to flatten to 1D array
    y_test = y_test_pd.to_numpy().ravel()

    layers = [9, 8, 4, 1]  # Example: Input layer, two hidden layers, and output layer
    nn_best = NeuralNet(
        layers=layers,
        epochs=200,
        learning_rate=0.1,
        momentum=0.2,
        fact="sigmoid",
        val_split=0.2
    )

    # Train on the full training set
    nn_best.fit(X_train, y_train)
    # Get the training and validation errors
    train_errors, val_errors = nn_best.loss_epochs()

    # Mean Square Error (MSE)
    test_mse = nn_best.compute_error(X_test, y_test)
    y_pred = nn_best.predict(X_test)

    # Ensure y_test and y_pred are 1D arrays
    y_test = np.ravel(y_test)
    y_pred = np.ravel(y_pred)

    # Mean Absolute Error (MAE)
    total_error = 0
    for y_predicted, y_real in zip(y_pred, y_test):
        total_error += abs(y_real - y_predicted)
    test_mae = total_error / len(y_test)

    # Mean Absolute Percentage Error (MAPE)
    total_error = 0
    epsilon = 1e-7  # Avoid division by zero
    for y_predicted, y_real in zip(y_pred, y_test):
        if y_real != 0:
            total_error += abs((y_real - y_predicted) / (y_real + epsilon))
    test_mape = (total_error / len(y_test)) * 100

    print(f"\nFinal Model Performance on Test Set:")
    print(f"MSE: {test_mse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"MAPE: {test_mape:.4f}%")



