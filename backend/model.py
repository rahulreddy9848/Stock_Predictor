import yfinance as yf
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import os
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TickerDataError(Exception):
    pass

class StockPredictor:
    def __init__(self, ticker='AAPL', look_back=60, epochs=3, batch_size=1):
        self.ticker = ticker
        self.look_back = look_back
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.df = None
        self.train_len = 0

    def load_data(self, period='10y'):
        try:
            data = yf.download(self.ticker, period=period, auto_adjust=False, group_by="ticker")
        except Exception as e:
            raise TickerDataError(f"Failed to download data for ticker '{self.ticker}': {e}")
        
        if isinstance(data.columns, pd.MultiIndex):
            self.df = data[self.ticker][['Adj Close']].copy()
        else:
            self.df = data[['Adj Close']].copy()
        
        self.df.rename(columns={'Adj Close': 'Close'}, inplace=True)
        
        if self.df.empty:
            raise TickerDataError(f"No data found for ticker: {self.ticker}. It may be invalid or delisted.")

        self.train_len = math.ceil(len(self.df[['Close']].values) * 0.8)
        return self.df

    def preprocess_data(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        prices = self.df[['Close']].values

        self.scaler.fit(prices.reshape(-1, 1)[:self.train_len, :])
        scaled_prices = self.scaler.transform(prices.reshape(-1, 1))

        train_data = scaled_prices[0:self.train_len, :]
        x_train, y_train = [], []
        for i in range(self.look_back, len(train_data)):
            x_train.append(scaled_prices[i - self.look_back:i, 0])
            y_train.append(scaled_prices[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        test_data = scaled_prices[self.train_len - self.look_back:, :]
        x_test = []
        y_test = prices[self.train_len:]
        for i in range(self.look_back, len(test_data)):
            x_test.append(test_data[i - self.look_back:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        return x_train, y_train, x_test, y_test

    def build_model(self):
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=(self.look_back, 1)))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        self.model = model
        return model

    def train_model(self, x_train, y_train):
        if self.model is None:
            self.build_model()
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs)

    def predict(self, x_test):
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train_model() or load_model() first.")
        predictions = self.model.predict(x_test)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions

    def save_model(self, path=None):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        if path is None:
            path = os.path.join("data", f'{self.ticker}_stock_prediction_model.h5')
        self.model.save(path)

    def save_scaler(self, path=None):
        if path is None:
            path = os.path.join("data", f'{self.ticker}_scaler.joblib')
        joblib.dump(self.scaler, path)

    def load_trained_model(self, path=None):
        if path is None:
            path = os.path.join("data", f'{self.ticker}_stock_prediction_model.h5')
        self.model = load_model(path)

    def load_scaler(self, path=None):
        if path is None:
            path = os.path.join("data", f'{self.ticker}_scaler.joblib')
        self.scaler = joblib.load(path)

    def get_last_n_days_scaled_data(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        prices = self.df[['Close']].values

        last_n_days = prices[-self.look_back:]
        scaled_last_n_days = self.scaler.transform(last_n_days)
        return scaled_last_n_days

    def predict_future(self, days_to_predict=5):
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train_model() or load_model() first.")
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        future_predictions = []
        current_input = self.get_last_n_days_scaled_data()

        for _ in range(days_to_predict):
            input_pred = np.reshape(current_input, (1, self.look_back, 1))
            
            next_scaled_price = self.model.predict(input_pred)
            
            next_price = self.scaler.inverse_transform(next_scaled_price)[0][0]
            future_predictions.append(next_price)
            
            current_input = np.append(current_input[1:], next_scaled_price, axis=0)

        last_date = self.df.index[-1]
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days_to_predict + 1)]
        
        return pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions}).set_index('Date')

if __name__ == '__main__':
    predictor = StockPredictor(ticker='TCS.NS')
    predictor.load_data()
    x_train, y_train, x_test, y_test = predictor.preprocess_data()
    predictor.build_model()
    predictor.train_model(x_train, y_train)
    predictor.save_model()
    print("Model trained and saved.")

    new_predictor = StockPredictor(ticker='TCS.NS')
    new_predictor.load_data()
    new_predictor.load_trained_model()
    
    future_preds = new_predictor.predict_future(days_to_predict=5)
    print("Future predictions:")
    print(future_preds)
