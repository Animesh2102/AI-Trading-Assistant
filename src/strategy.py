def generate_signals(predictions, actual_prices, rsi_values):
    signals = []

    for i in range(len(predictions)):
        pred = predictions[i]
        price = actual_prices[i]
        rsi = rsi_values[i]

        if pred > price * 1.002 and rsi < 40:
            signals.append("BUY")
        elif pred < price * 0.998 and rsi > 60:
            signals.append("SELL")
        else:
            signals.append("HOLD")

    return signals