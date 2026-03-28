import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, seq_length=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []

    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i])
        y.append(scaled_data[i])

    return np.array(X), np.array(y), scaler