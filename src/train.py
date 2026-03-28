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
        loss = criterion(outputs, y_batch)

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
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

# Plot
plt.plot(y_test_actual, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.show()