import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, seq_length=20):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []

    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i])
        y.append(scaled_data[i][0])  # predict Close

    return np.array(X), np.array(y), scaler