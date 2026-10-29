def generate_signals(predictions, actual_prices, rsi_values):
    signals = []

    for i in range(1, len(predictions)):
        change = predictions[i] - predictions[i-1]
        rsi = rsi_values[i]

        # Hybrid logic (LSTM + RSI)
        if change > 0.2 and rsi < 45:
            signals.append("BUY")
        elif change < -0.2 and rsi > 55:
            signals.append("SELL")
        else:
            signals.append("HOLD")

    signals.insert(0, "HOLD")
    return signals