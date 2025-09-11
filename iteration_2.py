print("Initializing...")

import os
import math as m
import statistics as s
import datetime as dt
from dateutil.relativedelta import relativedelta
import tkinter as tk
from tkinter import messagebox, scrolledtext
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import threading
import sys

# -------------------------------
# Custom Log Frame (inside main GUI)
# -------------------------------
class RedirectText(object):
    def _init_(self, text_widget):
        self.output = text_widget

    def write(self, string):
        self.output.configure(state="normal")
        self.output.insert(tk.END, string)
        self.output.see(tk.END)
        self.output.configure(state="disabled")

    def flush(self):
        pass  # needed for sys compatibility

# Hide TensorFlow info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Try importing ML libraries
ML_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError:
    ML_AVAILABLE = False
    print("⚠ TensorFlow / scikit-learn not installed. Neural network features disabled.")

print("Libraries Successfully Ins")

# -------------------------------
# Safe input functions
# -------------------------------
def get_int(prompt, default):
    return default  # GUI will handle input

def get_float(prompt, default):
    return default  # GUI will handle input

# -------------------------------
# Helper function to build LSTM
# -------------------------------
def build_lstm_model(input_shape, units=50, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units // 2, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

# -------------------------------
# Main Analysis Function (unchanged)
# -------------------------------
def run_analysis(T, prd, p1, p2, seq_len, test_ratio, epochs, batch_size, predict_days):
    try:
        date_end = dt.date.today() - relativedelta(days=1)
        date_str_end = date_end.strftime("%d-%m-%y")
        date1y = dt.date.today() - relativedelta(years=1) + relativedelta(days=1)
        print(date1y.strftime("%d-%m-%y"), "to", date_str_end)

        # Download stock data
        D = yf.download(T, period=prd, progress=False)
        if D is None or D.empty:
            raise ValueError("No data downloaded. Check ticker symbol and internet connection.")

        # Handle DataFrame vs Series
        if "Close" in D.columns:
            close_series = D["Close"]
        elif ("Close", T) in D.columns:
            close_series = D[("Close", T)]
        else:
            raise KeyError("Could not find 'Close' prices in downloaded data.")

        CL = close_series.squeeze().tolist()
        xCL = [i + 1 for i in range(len(CL))]

        # Moving averages
        x1, x2, y1, y2 = [], [], [], []

        # Short MA
        for i in range(p1, len(CL) + 1):
            y1.append(s.mean(CL[i - p1 : i]))
            x1.append(i)

        # Long MA
        for i in range(p2, len(CL) + 1):
            y2.append(s.mean(CL[i - p2 : i]))
            x2.append(i)

        # Crossovers
        g_cr, d_cr = [], []
        s_idx = max(p1, p2)
        y1c, y2c = np.array(y1[s_idx - p1 :]), np.array(y2[s_idx - p2 :])
        cross_x = []
        if len(y1c) == len(y2c) and len(y1c) > 1:
            cross_idx_local = (np.where(np.diff(np.sign(y1c - y2c)))[0] + s_idx).tolist()
            for i in cross_idx_local:
                v1 = y1[i - p1] if (i - p1) < len(y1) else None
                v2 = y2[i - p2] if (i - p2) < len(y2) else None
                if v1 is None or v2 is None:
                    continue
                if v1 > v2:
                    g_cr.append((i, v1))
                else:
                    d_cr.append((i, v1))
            cross_x = cross_idx_local

        gx, gy = (list(x) for x in zip(*g_cr)) if g_cr else ([], [])
        dx, dy = (list(x) for x in zip(*d_cr)) if d_cr else ([], [])

        # -----------------------------
        # Plot 1: Moving averages
        # -----------------------------
        plt.figure(figsize=(11, 7))
        plt.plot(xCL, CL, color="blue", label="Closing Price")
        if y1:
            plt.plot(x1, y1, color="r", label=f"MA - {p1} Days")
        if y2:
            plt.plot(x2, y2, color="g", label=f"MA - {p2} Days")
        if gx:
            plt.scatter(gx, gy, marker="^", facecolor="green", label="Golden Cross", s=40, zorder=10)
        if dx:
            plt.scatter(dx, dy, marker="v", facecolor="black", label="Death Cross", s=40, zorder=10)

        plt.legend()
        plt.xlabel(f"Day count from {date1y}")
        plt.ylabel("Price")
        plt.title(f"{T} | MA({p1},{p2}) | upto {date_str_end}")
        plt.grid(True)
        plt.savefig(f"MAC {date_end} {T[:4]}")
        plt.show(block=False)

        # Profit Calculation
        by, sl = [], []
        sby, ssl = 0, 0
        rg = min(len(g_cr), len(d_cr))
        for i in range(rg):
            by.append(g_cr[i][0])
            sl.append(d_cr[i][0])
        for j in by:
            sby += CL[j]
        for k in sl:
            ssl += CL[k]
        prf = ssl - sby
        w = tk.Tk()
        w.title(f"Calculated Profits over {prd}")
        w.configure(bg="#e6f7ff")
        ot = f"📊 Period: {prd}\n📊 No of trading cycles: {rg}\n💰 Profit per share: {prf}"
        label = tk.Label(
            w,
            text=ot,
            font=("Arial", 12, "bold"),
            justify="center",
            bg="#e6f7ff",
            fg="#003366",
            padx=20,
            pady=20,
        )
        label.pack()
        w.update_idletasks()
        width = w.winfo_width()
        height = w.winfo_height()
        x = (w.winfo_screenwidth() // 2) - (width // 2)
        y = (w.winfo_screenheight() // 2) - (height // 2)
        x -= 120
        y -= 80
        w.geometry(f"{width}x{height}+{x}+{y}")
        w.lift()
        w.attributes("-topmost", True)
        w.after(200, lambda: w.attributes("-topmost", False))
        w.mainloop()

        # -----------------------------
        # Plot 2: Neural network part
        # -----------------------------
        if ML_AVAILABLE:
            prices = np.array(CL).reshape(-1, 1)
            scaler = MinMaxScaler((0, 1))
            prices_scaled = scaler.fit_transform(prices)

            def create_sequences(data, seq_length):
                X, y = [], []
                for i in range(seq_length, len(data)):
                    X.append(data[i - seq_length : i, 0])
                    y.append(data[i, 0])
                X, y = np.array(X), np.array(y)
                return X.reshape((X.shape[0], X.shape[1], 1)), y

            X, y = create_sequences(prices_scaled, seq_len)
            split = int(len(X) * (1 - test_ratio))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = build_lstm_model((seq_len, 1))
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

            pred_test = scaler.inverse_transform(model.predict(X_test))
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Future iterative predictions
            seq = prices_scaled[-seq_len:].flatten().tolist()
            future_preds_scaled = []
            for _ in range(predict_days):
                x_in = np.array(seq[-seq_len:]).reshape(1, seq_len, 1)
                pred_scaled = model.predict(x_in)
                future_preds_scaled.append(pred_scaled[0, 0])
                seq.append(pred_scaled[0, 0])
            future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

            # Plot test predictions
            plt.figure(figsize=(11, 6))
            plt.plot(range(len(CL)), CL, color="blue", label="Historical")
            start_idx = len(CL) - len(y_test_inv)
            plt.plot(range(start_idx, len(CL)), y_test_inv, color="orange", label="Actual present data")
            plt.plot(range(start_idx, len(CL)), pred_test, color="green", label="Predicted data")
            plt.legend()
            plt.title(f"{T} | LSTM Test Predictions")
            plt.xlabel(f"Day count from {date1y}")
            plt.ylabel("Price")
            plt.grid(True)
            plt.savefig(f"Running Tests {date_end} {T[:4]}")
            plt.show()

            # --- Realistic metrics + accuracy ---
            rmse = np.sqrt(mean_squared_error(y_test_inv, pred_test))
            r2 = r2_score(y_test_inv, pred_test)
            accuracy = max(0, r2) * 100

            # Create Tkinter popup for accuracy
            acc_win = tk.Tk()
            acc_win.title("📊 Model Evaluation")
            acc_win.configure(bg="#e6ffe6")

            ot = f"✅ Accuracy: {accuracy:.2f}%\n📌 RMSE: {rmse:.2f}\n📌 R² Score: {r2:.3f}"

            label = tk.Label(
                acc_win,
                text=ot,
                font=("Arial", 14, "bold"),
                justify="center",
                bg="#e6ffe6",
                fg="#003300",
                padx=20,
                pady=20,
            )
            label.pack()

            # Center window
            acc_win.update_idletasks()
            width = acc_win.winfo_width()
            height = acc_win.winfo_height()
            x = (acc_win.winfo_screenwidth() // 2) - (width // 2)
            y = (acc_win.winfo_screenheight() // 2) - (height // 2)
            x -= 120
            y -= 80
            acc_win.geometry(f"{width}x{height}+{x}+{y}")
            acc_win.lift()
            acc_win.attributes("-topmost", True)
            acc_win.after(200, lambda: acc_win.attributes("-topmost", False))

            # Plot future predictions
            plt.figure(figsize=(11, 6))
            plt.plot(xCL, CL, color="blue", label="Historical")
            plt.plot(range(len(CL), len(CL) + predict_days), future_preds, "--", color="red", label="Future Prediction")
            plt.legend()
            plt.xlabel(f"Day count from {date1y}")
            plt.ylabel("Price")
            plt.title(f"{T} | {predict_days}-day Forecast")
            plt.grid(True)
            plt.savefig(f"Future Predictions {date_end} {T[:4]}")
            plt.show()

        else:
            print("⚠ Skipping neural network part (TensorFlow not installed).")

    except Exception as e:
        print("Error occurred:", e)

# -------------------------------
# GUI Wrapper
# -------------------------------
def run_analysis_wrapper():
    try:
        ticker = e_ticker.get().upper().strip()
        period = e_period.get().strip()
        p1 = int(e_ma1.get())
        p2 = int(e_ma2.get())
        seq_len = int(e_seq.get())
        test_ratio = float(e_test.get())
        epochs = int(e_epochs.get())
        batch_size = int(e_batch.get())
        predict_days = int(e_pred.get())

        threading.Thread(
            target=run_analysis,
            args=(ticker, period, p1, p2, seq_len, test_ratio, epochs, batch_size, predict_days),
            daemon=True
        ).start()

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# -------------------------------
# Main GUI
# -------------------------------
root = tk.Tk()
root.title("📈 Stock Analyzer")
root.configure(bg="#f2f2f2")
root.lift()
root.attributes("-topmost", True)
root.after(200, lambda: root.attributes("-topmost", False))
root.focus_force()

labels = ["Ticker", "Period", "MA-1", "MA-2", "Seq Len", "Test Ratio", "Epochs", "Batch Size", "Predict Days"]
defaults = ["TSLA", "3y", "50", "200", "30", "0.2", "25", "16", "10"]
entries = []

for i, (lbl, dft) in enumerate(zip(labels, defaults)):
    tk.Label(root, text=lbl, bg="#f2f2f2", fg="#003366", font=("Arial", 10, "bold")).grid(row=i, column=0, padx=8, pady=4, sticky="e")
    ent = tk.Entry(root)
    ent.insert(0, dft)
    ent.grid(row=i, column=1, padx=8, pady=4)
    entries.append(ent)

e_ticker, e_period, e_ma1, e_ma2, e_seq, e_test, e_epochs, e_batch, e_pred = entries

btn = tk.Button(root, text="🚀 Run Analysis", command=run_analysis_wrapper, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
btn.grid(row=len(labels), column=0, columnspan=2, pady=10)

# -------------------------------
# Log Box in Main Window
# -------------------------------
log_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=12, state="disabled", bg="#1e1e1e", fg="white", font=("Consolas", 10))
log_text.grid(row=len(labels)+1, column=0, columnspan=2, padx=8, pady=8, sticky="nsew")

sys.stdout = RedirectText(log_text)
sys.stderr = RedirectText(log_text)

root.mainloop()
