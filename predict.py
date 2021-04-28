# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
col1, col2 = st.beta_columns(2)
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('HK Stock Analysis Web-App')

# # # # # # Disclaimer # # # # # 
st.header('Disclaimer: ')
st.write('You expressly agree that the use of this app/website is at your sole risk.')

st.write("The content of this webpage is not an investment advice and does not constitute any offer or solicitation to offer or recommendation of any investment product. It is for general purposes only and does not take into account your individual needs, investment objectives and specific financial circumstances. Investment involves risk.")

# # # # # # input box for stock number # # # # # 
selected_stock = st.text_input('Select Stock Number for Analysis and Prediction',"0010.HK")

# # # # # # Download Data # # # # # 
@st.cache(allow_output_mutation=True)
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)
name = yf.Ticker(selected_stock)

# # # # # # Chart of Stock # # # # # 
def plot_raw_data(selected_stock):
	fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                	open=data['Open'],
                	high=data['High'],
                	low=data['Low'],
                	close=data['Close'])])
	fig.layout.update(title_text=selected_stock+" "+name.info['longName'], xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

if not data.empty:
	plot_raw_data(selected_stock)
else:
	st.write('Please input a valid Hong Kong ticker!')
mov_data = data.set_index(pd.DatetimeIndex(data["Date"].values))
mov_day = st.selectbox("Enter number of days Moving Average:",
						(5,20,50,100,200),index=3)
# # # # # Moving Average # # # # #
mov_data["mov_avg"] = mov_data['Close'].rolling(window=int(mov_day),min_periods=0).mean()
str(mov_day), ' Days Moving Average of ', selected_stock
st.line_chart(mov_data[["mov_avg","Close"]])

# # # # # # Relative Strength Index (RSI) # # # # # 
st.header('Relative Strength Index (RSI)')
RSI_data = data.set_index(pd.DatetimeIndex(data["Date"].values))
delta = RSI_data["Adj Close"].diff(1)
delta = delta.dropna()
"Get the positive Gains (up) and the negative gains (down)"
up = delta.copy()
up[up<0]=0
down = delta.copy()
down[down>0]=0

time_period = 14
avg_gain = up.rolling(window=time_period).mean()
avg_loss = abs(down.rolling(window=time_period).mean())

"calculate RSI"
Rs = avg_gain / avg_loss
Rsi = 100.0 - (100.0 / (1.0 + Rs))
array_length = len(Rsi)
last_rsi = Rsi[array_length-1]

st.write('The RSI of the last Trading Day is ' + str(round(last_rsi,2)))

st.line_chart(Rsi, 800,250,use_container_width = True)

st.write("The indicator has an upper line, typically at 70, a lower line at 30, and a dashed mid-line at 50.")
# # # # # # # # # # # # # # # # # # # # # # # # # # 


# Predict forecast with Prophet. 
st.header("Prophet Prediction Model")
n_days = st.slider('Days of prediction:', 30, 120, 30, 30)
period = n_days

try:
	df_train = data[['Date','Close']]
	df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

	m = Prophet()
	m.fit(df_train)
	future = m.make_future_dataframe(periods=period)
	forecast = m.predict(future)
    
	st.write(f'Forecast of {selected_stock} plot for {n_days} days')
	fig1 = plot_plotly(m, forecast)

	st.plotly_chart(fig1,use_container_width = True)
except:
	pass





