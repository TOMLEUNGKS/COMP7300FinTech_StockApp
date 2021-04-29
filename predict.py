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
from datetime import datetime
from pandas_datareader.data import DataReader
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

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
info = yf.Ticker(selected_stock).info

 # # # # # # # # # #basic information # # # # # # # # # #
try:
	st.write("Company Name : " + info["longName"])
	st.write("Market Sector : " + info["sector"])
	st.write("Company website : " + info["website"])
	st.write("Previous Closing Price : " + str(info["previousClose"]))
	st.write("Regular Market Day High : " + str(info["regularMarketDayHigh"]))
	st.write("52 Weeks Change :" + str(info["52WeekChange"]))
	st.write("Quarterly Revenue Growth : " + str(info["earningsQuarterlyGrowth"]))
	st.write("Enterprise-Value-to-Revenue : " + str(info["enterpriseToRevenue"]))
except:
	pass
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

# # # # # # # # # Moving Average # # # # # # # # #
try:
	st.header("Moving Average ")
	mov_data = data.set_index(pd.DatetimeIndex(data["Date"].values))
	mov_day = st.selectbox("Enter number of days Moving Average:",
						(5,20,50,100,200),index=3)

	mov_data["mov_avg"] = mov_data['Close'].rolling(window=int(mov_day),min_periods=0).mean()
	str(mov_day), ' Days Moving Average of ', selected_stock
	st.line_chart(mov_data[["mov_avg","Close"]])
except: 
	pass
# # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # comparing the stock with others# # # # # # # # # # # # # # # # # #
try:
	st.header("Fundamental Analysis")
	compare_stock = st.text_input('Select Stock that you wish to compare with',"0005.HK")
	tickers = [selected_stock, compare_stock]
	c_info = yf.Ticker(compare_stock).info
	compare_info = []
	compare_info.append(c_info)
	compare_info.append(info)

	st.write("Comparing **" + info["longName"] + "** with **" + c_info["longName"] + "**")

	fundanentals = ["trailingAnnualDividendYield", "marketCap", "beta", "forwardPE"]
	df = pd.DataFrame(compare_info)
	df = df.set_index('shortName')

	st.write("Trailing Dividend Yield")
	st.bar_chart(df.trailingAnnualDividendYield, 1000,250,use_container_width = True)

	st.write("Market Capitalization")
	st.bar_chart(df.marketCap.transpose(), 1000,250,use_container_width = True)

	st.write("Beta")
	st.bar_chart(df.beta, 1000,250,use_container_width = True)

	st.write("Forward price-to-earnings (P/E) ratio")
	st.bar_chart(df.forwardPE, 1000,250,use_container_width = True)
except:
	pass
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



# # # # # # Relative Strength Index (RSI) # # # # # 

st.header('Relative Strength Index (RSI)')
try:
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
except: 
	pass
# # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # Predict forecast with Prophet. # # # # # # # # # # # # # 

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
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



# # # # # # # # # # # # # # # # Prediction using LSTM# # # # # # # # # # # # # # # # # 
st.header("Prediction with Long Short-term Memory (LSTM)")
try:
	# Create a new dataframe with only the 'Close column 
	LSTM_df = DataReader(selected_stock, data_source='yahoo', start='2019-03-01', end=datetime.now())
	LSTM_data = LSTM_df.filter(['Close'])

	# Convert the dataframe to a numpy array
	LSTM_dataset = LSTM_data.values
	# Get the number of rows to train the model on
	training_data_len = int(np.ceil( len(LSTM_dataset) * .95 ))

	# Scale the data
	# from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler(feature_range=(0,1))
	scaled_data = scaler.fit_transform(LSTM_dataset)

	# Create the training data set 
	# Create the scaled training data set
	train_data = scaled_data[0:int(training_data_len), :]
	# Split the data into x_train and y_train data sets
	x_train = []
	y_train = []

	print(len(train_data))
	for i in range(30, len(train_data)):
		x_train.append(train_data[i-30:i, 0])
		y_train.append(train_data[i, 0])

		# if i<= 61:
		#     print(x_train)
		#     print(y_train)
		#     print()
		
	# Convert the x_train and y_train to numpy arrays 
	x_train, y_train = np.array(x_train), np.array(y_train)

	# Reshape the data
	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
	# x_train.shape

	# from keras.models import Sequential
	# from keras.layers import Dense, LSTM

	# Build the LSTM model
	model = Sequential()
	model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
	model.add(LSTM(64, return_sequences=False))
	model.add(Dense(25))
	model.add(Dense(1))

	# Compile the model
	model.compile(optimizer='adam', loss='mean_squared_error')

	# Train the model
	model.fit(x_train, y_train, batch_size=1, epochs=1)

	# Create the testing data set
	# Create a new array containing scaled values from index 1543 to 2002 
	test_data = scaled_data[training_data_len - 60: , :]
	# Create the data sets x_test and y_test
	x_test = []
	y_test = LSTM_dataset[training_data_len:, :]
	for i in range(60, len(test_data)):
		x_test.append(test_data[i-60:i, 0])
		
	# Convert the data to a numpy array
	x_test = np.array(x_test)

	# Reshape the data
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

	# Get the models predicted price values 
	predictions = model.predict(x_test)
	predictions = scaler.inverse_transform(predictions)

	# Get the root mean squared error (RMSE)
	rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
	# rmse

	# Plot the data
	train = data[:training_data_len]
	valid = data[training_data_len:]
	# valid['Predictions'] = predictions
	predictions = valid.iloc[:,1]


	# Visualize the data
	st.write(f'Forecast of {selected_stock} using LSTM')
	plt.figure(figsize=(16,8))
	plt.xlabel('Date', fontsize=18)
	plt.ylabel('Close Price HKD ($)', fontsize=18)
	plt.plot(train['Close'])
	plt.plot(predictions)
	plt.legend(['Train', 'Predictions'], loc='lower right')
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()
	# plt.show()
	# st.line_chart(mov_data[["mov_avg","Close"]])
	# g = Prophet()
	# g.fit(train['Close'])
	# # future = m.make_future_dataframe(periods=period)
	# p = Prophet().fit(predictions)
	# st.write(f'Forecast of {selected_stock} using LSTM')
	# fig2 = plot_plotly(g,p)
	# st.plotly_chart(plt,use_container_width = True)



except:
	pass