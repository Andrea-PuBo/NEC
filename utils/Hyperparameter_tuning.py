#%% md
# # Hyperparameter tuning for neural network
# 
# > Back Propagation Implementation - December 2024
# >
# > *Andrea Pujals Bocero*
# >
# > NEC First Assignment - Universitat Rovira i Virgili
# 
# This notebook explores different combinations of hyperparameters to identify the best configuration for a neural network. 
# We will evaluate at least 10 combinations of the following hyperparameters:
# 
# - **Number of layers**
# - **Layer structure**
# - **Number of epochs**
# - **Learning rate**
# - **Momentum**
# - **Activation function**
# 
# Metrics evaluated:
# - **Mean Absolute Percentage Error (MAPE)**
# - **Mean Absolute Error (MAE)**
# - **Mean Squared Error (MSE)**
# 
# Plots included:
# - Scatter plots of predicted vs true values for representative combinations.
# - Training and validation loss evolution over epochs.
# 
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.NeuralNet import NeuralNet  # Import my neural network implementation
import itertools
#%%
#read and parse the .csv features file for A1-turbine normalized data
df = pd.read_csv('A1-synthetic.csv', decimal=".")
df.describe()
#%%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled.describe()
#%%
columns = df_scaled.shape[1]

input_columns = df_scaled.columns[0 : 9] # Select the first 9 columns
features = df_scaled[input_columns].values

output_column = df_scaled.columns[9] # Select the 10th column (index 9) as the target
targets = df_scaled[output_column].values

print(features.shape)
print(targets.shape)
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2, random_state= 42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#%%
def hyperparameter_search(X_train, y_train, X_val, y_val, param_grid):
    """
    Perform hyperparameter search for the given neural network class.
    """
    param_combinations = list(itertools.product(*param_grid.values()))
    results = []

    for params in param_combinations:
        layers, epochs, lr, momentum, activation = params
        
        # Initialize and train the network
        nn = NeuralNet(layers, epochs, lr, momentum, activation, val_split=0.2)
        # Call the fit method
        nn.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = nn.predict(X_val)
        mse = ((y_val - y_pred) ** 2).mean()
        mae = abs(y_val - y_pred).mean()
        epsilon = 1e-7  # A very small value to avoid division by zero
        mape = (abs((y_val - y_pred) / (y_val + epsilon))).mean() * 100


        results.append({
            "Layers": layers,
            "Epochs": epochs,
            "Learning Rate": lr,
            "Momentum": momentum,
            "Activation": activation,
            "MSE": mse,
            "MAE": mae,
            "MAPE": mape
        })

    return pd.DataFrame(results)
#%%
# Define hyperparameter grid
param_grid = {
    "layers": [
        #[9, 8, 4, 1],  # Simple
        #[9, 16, 8, 1]  # Moderate
        [9, 16, 16, 8, 1]  # Two layers with 16 units

    ],
    "epochs": [100, 500],
    "learning_rate": [0.01, 0.001],
    "momentum": [0.5, 0.9],
    "activation": ["tanh"]  # Activation for hidden layers
}
#%%
results = hyperparameter_search(X_train, y_train, X_test, y_test, param_grid)

# Sort and display results
results_sorted = results.sort_values(by="MSE", ascending=True)
print(results_sorted)
#%%
# Best configuration
best_params = results.loc[results["MSE"].idxmin()]
print("Best Parameters:", best_params)

# Scatter plot for best configuration
nn_best = NeuralNet(
    best_params["Layers"],
    best_params["Epochs"],
    best_params["Learning Rate"],
    best_params["Momentum"],
    best_params["Activation"],
    val_split=0.2
)
nn_best.fit(X_train, y_train)
y_best_pred = nn_best.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_best_pred)
plt.title("Predicted vs. True Values for Best Configuration")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.show()

# Loss plot for best configuration
train_losses, val_losses = nn_best.loss_epochs()
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_losses, label="Training Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training and Validation Loss Evolution")
plt.legend()
plt.show()