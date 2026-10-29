def backtest(signals, prices, initial_balance=100000):
    balance = initial_balance
    shares = 0

    for i in range(len(signals)):
        signal = signals[i]
        price = prices[i]

        if signal == "BUY" and balance > price:
            shares = balance // price
            balance -= shares * price

        elif signal == "SELL" and shares > 0:
            balance += shares * price
            shares = 0

    final_value = balance + shares * prices[-1]
    profit = final_value - initial_balance

    return final_value, profit