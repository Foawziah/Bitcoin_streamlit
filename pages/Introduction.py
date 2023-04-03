import streamlit as st
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
        body {{
            color: black;
            font-family: 'Helvetica Neue', Helvetica, sans-serif;
            font-size: 16px;
        }}
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover;
            background-position: center;
        }}
        .st-bb {{
            border-radius: 5px;
            box-shadow: 0px 0px 5px 0px rgba(0,0,0,0.2);
            padding: 20px;
            margin: 20px;
        }}
        h1 {{
            text-align: center;
            color: blue;
            font-size: 36px;
            font-weight: bold;
        }}
        h2 {{
            font-size: 32px;
        }}
        ul {{
            margin: 30px 0;
            padding-left: 20px;
            color: black;
            font-size: 32px;
        }}
        li {{
            margin-bottom: 0px;
            font-weight: bold;
            font-size:17px;
        }}
        .fontli{{
            font-size: 25px;
        }}
    </style>
    """,
    unsafe_allow_html=True
    )

add_bg_from_local('money-2.jpg')

# Set the title of the app
st.title("Bitcoin Price Prediction")

# Use Markdown syntax to write a bold and blue header
st.markdown("""<h1>Models Used</h1><h2 style='color:black'>Overview</h2>
<ul>
    <li class='fontli'>Baseline sentiment analysis</li>
    <li class='fontli'>RandomForestRegressor</li>
    <li class='fontli'>Recurrent Neural Network</li>
    <li class='fontli'> XGBRegressor model</li>
    <li class='fontli'>Time Series forecasting with Prophet</li>
    <li class='fontli'>LinearRegression</li>
    <li class='fontli'>Natural Language Processing (NLP)</li>
    <li class='fontli'>OLS Regression between sentiment score and price change</li>
    <li class='fontli'>Poisson Regression</li>
    <li class='fontli'>Fine-tuning BERT</li>
    <li class='fontli'>MultinomialNB</li>
</ul>

""", unsafe_allow_html=True)


