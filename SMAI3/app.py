import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import tensorflow as tf
import matplotlib
import sklearn
import pickle
st.title("Stock Price Visualization")

days = st.sidebar.slider("Select number of days", min_value=1, max_value=60, value=60)

st.sidebar.subheader("Select Stock(s)")
selected_stocks = st.sidebar.text_input("Enter stock symbols separated by commas (e.g., AAPL,MSFT)", value="AAPL")
stocks = [s.strip().upper() for s in selected_stocks.split(',')]
value_variable = selected_stocks

def plot_stock_price(ticker_symbol, days):
    data = yf.download(ticker_symbol, period=f"{days}d")
    plt.figure(figsize=(10, 6))
    plt.plot(data['Adj Close'])
    plt.title(f"{ticker_symbol} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    st.pyplot(plt)


if stocks:
    for stock in stocks:
        plot_stock_price(stock, days)

from pandas_datareader import data as pdr
from datetime import datetime
yf.pdr_override()
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
tech_list.extend(stocks)
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
for stock in tech_list:
    globals()[stock] = yf.download(stock, start, end)        
df = pdr.get_data_yahoo(value_variable, start='2012-01-01', end=datetime.now())
data = df.filter(['Close'])
dataset = data.values
training_data_len = int(np.ceil( len(dataset) * .95 ))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
train_data = scaled_data[0:int(training_data_len), :]
train_data_df = pd.DataFrame(train_data) 


#st.write("Обучающий набор данных:", train_data_df)
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense, LSTM

model = pickle.load(open('C:/Users/Kisliy/Desktop/SMAI3/models/model.pkl','rb'))


test_data = scaled_data[training_data_len - 60: , :]
future_days = 60
x_future = []
last_60_days = scaled_data[-60:]
x_test = []
y_test = dataset[training_data_len:, :]


for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    x_future.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
#predictions[:10]

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
# r2 = r2_score(y_test, predictions)
# st.write('MSE: ', mse)
# st.write('RMSE: ', rmse)
# st.write('R2_score: ', r2)

train = data[:training_data_len].copy()
valid = data[training_data_len:].copy()
valid.loc[:, 'Прогнозы'] = predictions


st.set_option('deprecation.showPyplotGlobalUse', False)


plt.figure(figsize=(16, 6))
plt.title('Модель')
plt.xlabel('Дата', fontsize=18)
plt.ylabel('Цена закрытия в USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Прогнозы']])
plt.legend(['Обучение', 'Валидация', 'Прогнозы'], loc='lower right')
st.pyplot()

x_future = np.array(x_future)
x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))

future_predictions = model.predict(x_future)
future_predictions = scaler.inverse_transform(future_predictions)

future_dates = pd.date_range(df.index[-1], periods=future_days+1, freq='B')[1:]

future_df = pd.DataFrame(index=future_dates, columns=['Future Close'])
future_df.index.name = 'Date'
future_df['Future Close'] = future_predictions[:60].flatten()

mse = mean_squared_error(y_test, predictions)
#st.write('Current MSE:', mse)

# while True:
#     model.fit(x_train, y_train)
#     future_predictions = model.predict(x_future)
#     future_predictions = scaler.inverse_transform(future_predictions)

#     mse_future = mean_squared_error(y_test, future_predictions)

#     if mse_future < 15:
#         st.write('Current MSE:', mse_future)
#         print('Current MSE:', mse_future)
#         break
#     else:
#         st.write('Retraining the model...')
#         print('Current MSE:', mse_future)
        
future_df['Future Close'] = future_predictions[:60].flatten()
selected_days = days
future_df = pd.DataFrame(index=future_dates[:selected_days], columns=['Future Close'])
future_df.index.name = 'Date'
future_df['Future Close'] = future_predictions[:selected_days].flatten()
final_df = pd.concat([df['Close'], future_df['Future Close']])

st.title("Stock Price Visualization")

plt.figure(figsize=(16, 6))
plt.title('Future Close Price Prediction')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price in USD ($)', fontsize=18)

plt.plot(final_df, label='Actual Data')
plt.axvline(x=df.index[-1], color='r', linestyle='--', label='End of Actual Data')
plt.plot(future_df['Future Close'], label='Future Predictions', linestyle='-')

plt.legend()
st.pyplot()
