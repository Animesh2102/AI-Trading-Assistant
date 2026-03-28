import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_loader import fetch_stock_data
from preprocess import create_sequences
from model import LSTMModel

# Load data
df = fetch_stock_data()
data = df.values

# Preprocess
X, y, scaler = create_sequences(data)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# Model
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10

for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

import matplotlib.pyplot as plt

# Convert test data
X_test = torch.tensor(X_test, dtype=torch.float32)

model.eval()
predictions = model(X_test).detach().numpy()

# Inverse scale
import numpy as np

# Create dummy array with 4 features
dummy_pred = np.zeros((predictions.shape[0], 4))
dummy_pred[:, 0] = predictions[:, 0]

dummy_actual = np.zeros((y_test.shape[0], 4))
dummy_actual[:, 0] = y_test

# Inverse transform
predictions = scaler.inverse_transform(dummy_pred)[:, 0]
y_test_actual = scaler.inverse_transform(dummy_actual)[:, 0]

# After Prediction Analysis
from strategy import generate_signals

# Use last part of RSI (aligned with test data)
rsi_values = df['RSI'].values[-len(predictions):]

signals = generate_signals(predictions, y_test_actual, rsi_values)

print(signals[:20])

#Backtest

from backtest import backtest

final_value, profit = backtest(signals, y_test_actual)

print("Final Value:", final_value)
print("Profit:", profit)

# Plot
plt.plot(y_test_actual, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.show()