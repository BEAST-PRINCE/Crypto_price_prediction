import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error


class CryptoPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crypto Price Predictor")
        self.root.geometry("900x700")
        
        # Initialize model and data attributes
        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.create_widgets()

    def create_widgets(self):
        # Main frame for controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky='ew')
        self.root.columnconfigure(0, weight=1)
        
        # Date selection
        ttk.Label(control_frame, text="Start Date:").grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.start_date = DateEntry(control_frame, width=12, background='darkblue', foreground='white', borderwidth=2, 
                                   year=2022)
        self.start_date.grid(row=0, column=1, padx=10, pady=10)

        # Crypto selector
        ttk.Label(control_frame, text="Cryptocurrency:").grid(row=1, column=0, padx=10, pady=10, sticky='w')
        self.crypto_var = tk.StringVar()
        self.crypto_menu = ttk.Combobox(control_frame, textvariable=self.crypto_var, 
                                       values=["BTC", "ETH", "DOGE", "XRP", "ADA", "SOL", "DOT"], state="readonly")
        self.crypto_menu.current(0)
        self.crypto_menu.grid(row=1, column=1, padx=10, pady=10)

        # Currency selector
        ttk.Label(control_frame, text="Currency:").grid(row=2, column=0, padx=10, pady=10, sticky='w')
        self.currency_var = tk.StringVar()
        self.currency_menu = ttk.Combobox(control_frame, textvariable=self.currency_var, 
                                         values=["USD", "INR", "EUR", "GBP", "JPY"], state="readonly")
        self.currency_menu.current(0)
        self.currency_menu.grid(row=2, column=1, padx=10, pady=10)

        # Model parameters frame
        model_frame = ttk.LabelFrame(control_frame, text="Model Parameters")
        model_frame.grid(row=0, column=2, rowspan=3, padx=20, pady=10, sticky='ns')
        
        ttk.Label(model_frame, text="Prediction Days:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.prediction_days_var = tk.IntVar(value=10)
        self.prediction_days_entry = ttk.Spinbox(model_frame, from_=5, to=60, textvariable=self.prediction_days_var, width=8)
        self.prediction_days_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Epochs:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.epochs_var = tk.IntVar(value=10)
        self.epochs_entry = ttk.Spinbox(model_frame, from_=1, to=100, textvariable=self.epochs_var, width=8)
        self.epochs_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Batch Size:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.batch_size_var = tk.IntVar(value=32)
        self.batch_size_entry = ttk.Spinbox(model_frame, from_=8, to=128, textvariable=self.batch_size_var, width=8)
        self.batch_size_entry.grid(row=2, column=1, padx=5, pady=5)

        # Buttons frame
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=1, column=0, pady=10)
        
        self.predict_button = ttk.Button(button_frame, text="Predict", command=self.predict)
        self.predict_button.grid(row=0, column=0, padx=10)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(button_frame, textvariable=self.status_var, font=("Arial", 10))
        self.status_label.grid(row=0, column=1, padx=20)

        # Results frame
        result_frame = ttk.LabelFrame(self.root, text="Prediction Results")
        result_frame.grid(row=2, column=0, padx=10, pady=10, sticky='ew')
        
        self.result_label = ttk.Label(result_frame, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)

        # Graph frame
        graph_frame = ttk.Frame(self.root)
        graph_frame.grid(row=3, column=0, padx=10, pady=10, sticky='nsew')
        self.root.rowconfigure(3, weight=1)
        
        self.figure = plt.Figure(figsize=(7, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def fetch_data(self, crypto, currency, start_date, end_date):
        """Fetch cryptocurrency data using yfinance"""
        self.status_var.set("Downloading data from Yahoo Finance...")
        self.root.update()
        
        # Try different ticker patterns for Yahoo Finance
        ticker_patterns = [
            f"{crypto}-{currency}",    # Standard format
            f"{crypto}{currency}=X",   # Alternative format for some pairs
            f"{crypto}-{currency}=X"   # Another alternative
        ]
        
        data = None
        for ticker in ticker_patterns:
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                if not data.empty:
                    self.status_var.set("Data downloaded successfully")
                    break
            except Exception:
                continue
                
        if data is None or data.empty:
            raise Exception(f"No data available for {crypto}-{currency}. Try a different pair.")
            
        return data

    def predict(self):
        try:
            # Update UI to show processing
            self.predict_button.config(state=tk.DISABLED)
            self.status_var.set("Processing...")
            self.root.update()
            
            # Get parameters
            crypto = self.crypto_var.get()
            currency = self.currency_var.get()
            prediction_days = self.prediction_days_var.get()
            epochs = self.epochs_var.get()
            batch_size = self.batch_size_var.get()
            start_date = self.start_date.get_date()
            end_date = dt.datetime.now()
            
            # Fetch data
            try:
                self.data = self.fetch_data(crypto, currency, start_date, end_date)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.predict_button.config(state=tk.NORMAL)
                self.status_var.set("Ready")
                return
            
            # Check if we have enough data
            if self.data.empty or len(self.data) < prediction_days * 2:
                messagebox.showerror("Error", f"Not enough data to train model. Need at least {prediction_days * 2} days of data.")
                self.predict_button.config(state=tk.NORMAL)
                self.status_var.set("Ready")
                return

            # Preprocess data
            scaled_data = self.scaler.fit_transform(self.data['Close'].values.reshape(-1, 1))

            x_train, y_train = [], []

            for i in range(prediction_days, len(scaled_data)):
                x_train.append(scaled_data[i - prediction_days:i, 0])
                y_train.append(scaled_data[i, 0])

            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            # Build and train the model
            self.status_var.set("Training model...")
            self.root.update()
            
            model = Sequential()
            # First LSTM layer with return sequences
            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(Dropout(0.2))
            # Second LSTM layer
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            # Dense output layer
            model.add(Dense(units=1))

            model.compile(optimizer='adam', loss='mean_squared_error')
            
            model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
            
            # Prepare data for prediction
            actual_prices = self.data['Close'].values
            
            # Split data for testing
            train_len = int(len(scaled_data) * 0.8)
            test_data = scaled_data[train_len - prediction_days:]
            
            # Create test datasets
            x_test = []
            y_test = actual_prices[train_len:]
            
            for i in range(prediction_days, len(test_data)):
                x_test.append(test_data[i - prediction_days:i, 0])
            
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            
            # Make predictions
            predicted_prices = model.predict(x_test)
            predicted_prices = self.scaler.inverse_transform(predicted_prices)
            
            # Calculate metrics
            if len(predicted_prices) > 0 and len(y_test) >= len(predicted_prices):
                mse = mean_squared_error(y_test[:len(predicted_prices)], predicted_prices)
                mae = mean_absolute_error(y_test[:len(predicted_prices)], predicted_prices)
                accuracy = 100 - (mae / np.mean(y_test[:len(predicted_prices)]) * 100)
            else:
                mse, mae, accuracy = 0, 0, 0
            
            # Predict next day
            next_day_input = scaled_data[-prediction_days:]
            next_day_input = np.reshape(next_day_input, (1, prediction_days, 1))
            next_day_pred = model.predict(next_day_input)
            next_day_price = self.scaler.inverse_transform(next_day_pred)[0][0]
            
            # Update UI
            self.ax.clear()
            
            # Plot training data
            train_dates = self.data.index[:train_len]
            train_prices = actual_prices[:train_len]
            self.ax.plot(train_dates, train_prices, color='blue', label='Training Data')
            
            # Plot actual test data
            test_dates = self.data.index[train_len:]
            test_prices = actual_prices[train_len:]
            self.ax.plot(test_dates, test_prices, color='black', label='Actual Price')
            
            # Plot predicted prices
            if len(test_dates) >= len(predicted_prices):
                prediction_dates = test_dates[:len(predicted_prices)]
                self.ax.plot(prediction_dates, predicted_prices, color='green', label='Predicted Price')
            
            # Add next day prediction marker
            next_day = self.data.index[-1] + dt.timedelta(days=1)
            self.ax.scatter(next_day, next_day_price, color='red', s=50, label='Next Day Prediction')
            
            self.ax.legend()
            self.ax.set_title(f'{crypto}-{currency} Price Prediction')
            self.figure.autofmt_xdate()  # Rotate date labels for better readability
            self.canvas.draw()
            
            # Update result label
            current_price = actual_prices[-1]
            price_change = next_day_price - current_price
            price_change_percent = (price_change / current_price) * 100
            
            self.result_label.config(text=(
                f"Current price: {current_price:.2f} {currency}\n"
                f"Predicted next price: {next_day_price:.2f} {currency} "
                f"({'↑' if price_change >= 0 else '↓'}{abs(price_change):.2f}, {abs(price_change_percent):.2f}%)\n"
                f"Model accuracy: {accuracy:.2f}% | MSE: {mse:.2f} | MAE: {mae:.2f}"
            ))
            
            self.status_var.set("Prediction complete")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.predict_button.config(state=tk.NORMAL)


if __name__ == '__main__':
    root = tk.Tk()
    app = CryptoPredictorApp(root)
    root.mainloop()