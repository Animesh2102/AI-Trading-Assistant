def backtest(signals, prices, initial_balance=100000):
    balance = initial_balance
    shares = 0
    buy_price = 0  # NEW: track entry price

    for i in range(len(signals)):
        signal = signals[i]
        price = prices[i]
        transaction_cost = 0.001  # 0.1%

        # BUY
        if signal == "BUY" and balance > price and shares == 0:
            shares = (balance * 0.5) // price
            balance -= shares * price * (1 + transaction_cost)
            buy_price = price
            print(f"BUY at {price}")

        # STOP LOSS (3%)
        elif shares > 0 and price < buy_price * 0.97:
            balance += shares * price * (1 - transaction_cost)
            print(f"STOP LOSS SELL at {price}")
            shares = 0

        # TAKE PROFIT (5%)
        elif shares > 0 and price > buy_price * 1.05:
            balance += shares * price * (1 - transaction_cost)
            print(f"TAKE PROFIT SELL at {price}")
            shares = 0

        # NORMAL SELL
        elif signal == "SELL" and shares > 0:
            balance += shares * price * (1 - transaction_cost)
            print(f"SELL at {price}")
            shares = 0

    final_value = balance + shares * prices[-1]
    profit = final_value - initial_balance

    return final_value, profit