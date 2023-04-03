import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from PIL import Image

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
add_bg_from_local('Afghan-Coins.jpg')
st.markdown("""<h1 style='color:black'>Stock line chart</h1>""", unsafe_allow_html=True)

#st.title('Stock line chart')
st.write('**starting** the *build* of `Bitcoin app` :penguin :smile:')
st.write('data is taken from[Bitcoin](https://www.binance.com/en/markets/overview')
st.header('Data')
df = pd.read_csv('Binance_BTCUSDT_d.csv')
st.write('Display a sample of 20 datapoints', df.sample(20))
# Create figure with multiple traces
fig = go.Figure()
for col in ['Open', 'High', 'Low', 'Close']:
    fig.add_trace(go.Scatter(x=df['Date'], y=df[col], name=col))

# Update layout
fig.update_layout(
    title='Stock Line Chart',
    xaxis_title='Date',
    yaxis_title='Price',
    legend_title='Price',
)

# Display plot in Streamlit
st.plotly_chart(fig)


slider_choice = st.sidebar.selectbox('You have the following options', ['yes', 'no'])
if slider_choice == 'yes':
    st.write('yes selected')
else:
    st.write('no selected')

data = st.sidebar.file_uploader('Upload data', type = ['csv'])

if data is not None:
    df = pd.read_csv(data)
    st.write(df)
    
file_imgs = st.sidebar.file_uploader('Input images', type=['png','jpg','jpeg'], accept_multiple_files=True)
if file_imgs is not None:
    for file_img in file_imgs:
        img = Image.open(file_img)
        st.image(img)
