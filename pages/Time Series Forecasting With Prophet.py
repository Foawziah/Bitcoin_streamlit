import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import streamlit as st
import datetime
import pytz
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('money-2.jpg')



# Set the color for the streamlit app to black
st.markdown("<style>h1,h2{color: black;}</style>", unsafe_allow_html=True)

# Load data
df = pd.read_csv('Binance_BTCUSDT_d.csv')

# Set the color for the pandas data frame to black
st.title('Bitcoin Prediction')
st.header('Data')

st.write('Display a sample of 20 datapoints', df.sample(20))

species = st.selectbox(f'Select Open', df.Open.unique())

# Change text color of dataframe
def highlight_text(s):
    return ['color: black']*len(s)  # Change 'black' to the desired color

df.style.apply(highlight_text, axis=1)






#st.bar_chart(df.groupby('Date')['Close'].count())
#st.map(df)

import streamlit as st
import pandas as pd

slider_choice = st.sidebar.selectbox('You have the following options', ['yes', 'no'])

if slider_choice == 'yes':
    st.write('yes selected')
else:
    st.write('no selected')

data = st.sidebar.file_uploader('Upload data', type=['csv'])

if data is not None:
    df = pd.read_csv(data)
    st.write(df)

    
#

# convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# select a date range
start = np.datetime64(datetime.datetime(2022, 6, 19, 0, 0, 0, 0, pytz.UTC).isoformat())
end = np.datetime64(datetime.datetime(2022, 6, 30, 0, 0, 0, 0, pytz.UTC).isoformat())

# get the weekly rows within the date range
weekly_rows = df[(df['Date'] >= start) & (df['Date'] <= end)].groupby([pd.Grouper(key='Date', freq='W-MON')]).first().reset_index()

# display the weekly rows in Streamlit
st.dataframe(weekly_rows)

# create the plot trace
trace1 = go.Scatter(
    x=weekly_rows['Date'],
    y=weekly_rows['Volume BTC'].astype(float),
    mode='lines',
    name='Bitcoin Price (Open)'
)

# create the plot layout
layout = go.Layout(
    title='Historical Bitcoin Volume (USD) (2022-2023) with the slider',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=12,
                     label='1y',
                     step='month',
                     stepmode='backward'),
                dict(count=36,
                     label='3y',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type='date'
    )
)

# create the plot figure
fig = go.Figure(data=[trace1], layout=layout)

# display the plot in Streamlit
st.plotly_chart(fig)


# Example data
dates = pd.date_range('2023-06-01', '2023-12-31', freq='H')
split_date = pd.Timestamp('2023-10-31')

# Convert the numpy array to a pandas Series with the same index as the original dataframe
df = pd.DataFrame(np.random.randn(len(dates)), index=dates, columns=['Weighted_Price'])

# Split the data into training and testing sets
data_train = df.loc[df.index <= split_date].copy()
data_test = df.loc[df.index > split_date].copy()

# Create a line plot using Matplotlib
fig, ax = plt.subplots(figsize=(15,5))
data_test \
    .rename(columns={'Weighted_Price': 'Test Set'}) \
    .join(data_train.rename(columns={'Weighted_Price': 'Training Set'}), how='outer') \
    .plot(ax=ax, style='', title='BTC Weighted_Price Price (USD) by Hours')
    
# Display the plot in Streamlit
st.pyplot(fig)
#


from prophet import Prophet
import pickle

# set the title of the Streamlit app with black color
st.markdown("<h1 style='color: black;'>Time Series Forecasting With Prophet</h1>", unsafe_allow_html=True)


# Load the data
data = pd.read_csv('Binance_BTCUSDT_d.csv')

# Rename columns
data = data.rename(columns={'Date': 'ds', 'Weighted_Price': 'y'})

# Add Weighted_Price column
data['Weighted_Price'] = (data['Open'] + data['Close'] + data['High'] + data['Low']) / 4

# Reorder columns
data = data[['ds', 'Weighted_Price', 'Open', 'High', 'Low', 'Close']]

# Convert to datetime
data['ds'] = pd.to_datetime(data['ds'])

# Filter to dates from 2019 onwards
data = data[data['ds'].dt.year >= 2019]

# Set the index
data = data.set_index('ds')

# Load the model from file
with open('prophet_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Filter to predictions from 2019 onwards
forecast = forecast[forecast['ds'].dt.year >= 2019]

# Plot the forecast
fig = go.Figure()

# Add the actual data
fig.add_trace(go.Scatter(x=data.index, y=data['Weighted_Price'], mode='lines', name='Actual'))

# Add the predicted data
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))

# Add the uncertainty interval
fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_upper'],
    fill=None,
    mode='lines',
    line_color='rgba(255,255,255,0)',
    showlegend=False,
))

fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_lower'],
    fill='tonexty',
    mode='lines',
    line_color='rgba(255,255,255,0)',
    showlegend=False,
))

# Set the layout
fig.update_layout(
    title='BTC/USDT Weighted Price Forecast',
    xaxis_title='Date',
    yaxis_title='Weighted Price',
    xaxis_rangeslider_visible=True,
)

# Show the plot
st.plotly_chart(fig)


from prophet.plot import plot_plotly, plot_components_plotly

# Make predictions on the test data
future = model.make_future_dataframe(periods=len(data_test), freq='D')
forecast = model.predict(future)

# Plot the forecast
fig = plot_plotly(model, forecast)

# Set the plot height and width
fig.update_layout(height=500, width=800)

# Plot the forecast components
fig_components = plot_components_plotly(model, forecast)

# Set the plot height and width
fig_components.update_layout(height=500, width=800)

# Display the plots in Streamlit
st.plotly_chart(fig)
st.plotly_chart(fig_components)
