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
X, y, scaler = create_sequences(data, seq_length=20)

# Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

# Model
model = LSTMModel(input_size=4)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 20

for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "model.pth")
print("Model saved!")