import streamlit as st
from datetime import datetime

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight") 

# For reading stock data from yahoo
from pandas_datareader.data import DataReader

START = "2015-01-01"
TODAY = datetime.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

input_stock = st.text_input("Please Input the stock that you interested", "0005.HK")

def load_stockdata(code):
    df = DataReader(code, data_source='yahoo', start='2019-03-01', end=datetime.now())
    return df


n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

data_load_state = st.text('Loading data...')
data = load_stockdata(input_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data)

# Plot raw data
def plot_raw_data():

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data["Date"], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data["Date"], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig) 

plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)