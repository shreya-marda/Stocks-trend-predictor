import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr        #to help us read data from any website
import datetime as dt
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = dt.datetime(2015,1,1)
end = dt.datetime(2023,3,15)

st.title("Stock Trend Predictor")

user_input = st.text_input("Enter stock ticker", "AAPL")  #for now we are keeping it default as apple "AAPL"

yf.pdr_override()
df = pdr.get_data_yahoo(user_input,start,end)

#Describing the data
st.subheader("Data from 2015 to 2022")
st.write(df.describe())

#Visualisation
st.subheader("Closing price vs Time chart ")
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

#plotting 100 days moving average
st.subheader("Closing price vs Time chart with 100 days moving average")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'g', label = "MA100")
plt.plot(df.Close)
plt.xlabel("Year")
plt.ylabel("Stock prices")
st.pyplot(fig)

#plotting 200 daya moving average
st.subheader("Closing price vs Time chart with 100 and 200 days moving average")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'g', label = "MA100")
plt.plot(ma200, 'r', label = "MA200")
plt.plot(df.Close)
plt.legend()
plt.xlabel("Year")
plt.ylabel("Stock prices")
st.pyplot(fig)

#splitting data into traning and testing
data_train = pd.DataFrame(df["Close"][0:int(len(df)*0.7)])
data_test = pd.DataFrame(df["Close"][int(len(df)*0.7):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

data_train_array = scaler.fit_transform(data_train)

#Load my model
model = load_model('keras_model.h5')

#Testing part
past_100_days = data_train.tail(100) 
final_df = past_100_days.append(data_test, ignore_index = True)
input_data = scaler.fit_transform(final_df)

X_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])
    
X_test, y_test = np.array(X_test), np.array(y_test)
y_predicted = model.predict(X_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0] #as we get an array here 
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final graph
st.subheader("Predictions vs Original")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'g', label = 'Original price')
plt.plot(y_predicted, 'r', label = 'Predicted price')
plt.xlabel('Time')
plt.ylabel('Stock price')
plt.legend()   #shows the labels for green and red lines
# plt.show() 
st.pyplot(fig2)

st.write("Original closing price for the last day", y_test[-1] * scale_factor)
st.write("Predicted closing price for the last day", y_predicted[-1] * scale_factor)
