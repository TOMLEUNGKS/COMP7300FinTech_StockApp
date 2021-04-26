# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
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
col1, col2 = st.beta_columns(2)
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')
st.write('Disclaimer: ')
st.write('You expressly agree that the use of this app/website is at your sole risk.')
selected_stock = st.text_input('Select dataset for prediction',"0005.HK")

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)


# Plot raw data
def plot_raw_data(selected_stock):
	fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                	open=data['Open'],
                	high=data['High'],
                	low=data['Low'],
                	close=data['Close'])])
	fig.layout.update(title_text=selected_stock, xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)


plot_raw_data(selected_stock)

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)

st.plotly_chart(fig1,use_container_width = True)






