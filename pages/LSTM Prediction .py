import pickle
import base64
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
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


def add_bg_from_local(file):
    with open(file, 'rb') as f:
        img_bytes = f.read()
    encoded_string = base64.b64encode(img_bytes)
    return f"""
    <style>
        .stApp {{
            background-image: url('data:image/png;base64,{encoded_string.decode()}');
            background-size: cover;
        }}
    </style>
    """
import torch
import torch.nn as nn
import numpy as np

#


# Define the LSTM model class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# Load the pre-trained model
def load_model():
    model = LSTM(input_dim=1, hidden_dim=32, output_dim=1, num_layers=2)
    model.load_state_dict(torch.load("my_model.pt"))
    return model

model = load_model()

# Define the Streamlit app
def main():
    st.title("LSTM Prediction")
    
    # Get user input
    num_predictions = st.slider("Select the number of predictions to make", 1, 100, 10)
    last_known_value = st.number_input("Enter the last known value", value=100.0, step=0.1)
    
    # Generate predictions
    with torch.no_grad():
        x = np.linspace(last_known_value, last_known_value + num_predictions, num_predictions).reshape(-1, 1, 1)
        x = torch.from_numpy(x).type(torch.Tensor)
        y_pred = model(x)
        y_pred = y_pred.numpy().flatten()
    
    # Show the predictions
    st.write("### Predictions:")
    st.line_chart(y_pred)

if __name__ == "__main__":
    main()

# Display an image from a URL

# Display an image from a local file

## OLS Regression between sentiment score and price change

# Display the saved plot
with open('bitcoin.html', 'r') as f:
    plot_html = f.read()
st.components.v1.html(plot_html, width=700, height=500)

with open('GRU.html', 'r') as f:
    plot_html = f.read()
st.components.v1.html(plot_html, width=700, height=500)
st.image('time-pred.png', width=850)
