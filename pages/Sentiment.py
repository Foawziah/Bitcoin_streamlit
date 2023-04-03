import streamlit as st
import pandas as pd
import plotly.express as px

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
#st.title('Bitcoin Sentiment Analysis')
st.markdown("""<h1 style='color:black'>Bitcoin Sentiment Analysis</h1>""", unsafe_allow_html=True)

# assume the data is already loaded and processed as in the original code
df = pd.read_csv('Binance_BTCUSDT_d.csv')

# convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# filter the data to only include dates between 2019 and 2023
start_date = '2019-01-01'
end_date = '2023-12-31'
df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# display the filtered dataset in Streamlit
st.dataframe(df)




# select a date range
start = pd.to_datetime('2022-06-19')
end = pd.to_datetime('2022-06-30')

# filter the data for the selected date range
date_range_data = df[(df['Date'] >= start) & (df['Date'] <= end)]

# group the data by week and calculate the sum of the Bitcoin Volume (USDT)
weekly_data = date_range_data.groupby(pd.Grouper(key='Date', freq='W-MON')).sum().reset_index()

# create the plot using Plotly Express
fig = px.line(weekly_data, x='Date', y='Volume USDT', title='Weekly Bitcoin Volume (USDT) for June 2022')

# display the plot in Streamlit
st.plotly_chart(fig)

# create the plot using Plotly Express with a slider for the date range
fig = px.line(df, x='Date', y='Volume USDT', title='Historical Bitcoin Volume (USDT)')
fig.update_layout(xaxis_range=[start, end])
fig.update_xaxes(rangeslider_visible=True)

# display the plot in Streamlit
st.plotly_chart(fig)

####
import streamlit as st
import plotly.graph_objs as go
import pandas as pd

df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

years = df.index.year.unique().tolist()

annotations_list = []
max_high = df['High'].max()

for year in years:
    
    df_aux = df.loc[df.index.year == year,]
    year_change = ((df_aux.iloc[-1]['Close'] - df_aux.iloc[0]['Open']) / df_aux.iloc[0]['Open']) * 100
    loc_x  = pd.to_datetime(str(year))
    loc_y  = df_aux['High'].values[0]/max_high + 0.05
    text   = '{:.1f}%'.format(year_change)
    
    annotation = dict(x=loc_x, y=loc_y,
                      xref='x', yref='paper',
                      showarrow=False, xanchor='center',
                      text=text)

    annotations_list.append(annotation)

candlestick = go.Candlestick(
                  x     = df.index,
                  open  = df.Open,
                  close = df.Close,
                  low   = df.Low,
                  high  = df.High
              )

layout = dict(
    width       = 800,
    height      = 350,
    title       = dict(text='<b>Bitcoin/USD yearly chart</b>', font=dict(size=25)),
    yaxis_title = dict(text='Price (USD)', font=dict(size=13)),
    margin      = dict(l=0, r=20, t=55, b=20),
    xaxis_rangeslider_visible = False,
    annotations = annotations_list
)

fig = go.Figure(data=[candlestick], layout=layout)

# select a date range
start = pd.to_datetime('2022-06-19')
end = pd.to_datetime('2022-06-30')

fig.add_shape(
            type="rect",
            x0=start,
            y0=0,
            x1=end,
            y1=max(df['High']),
            fillcolor="LightSalmon",
            opacity=0.5,
            layer="below",
            line=dict(width=0,)
)

st.plotly_chart(fig)








st.image('yearly.png', width=850)




#
st.image('future.png', width=850)
st.image('bit.png', width=850)


