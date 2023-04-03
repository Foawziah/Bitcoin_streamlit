
import pickle
import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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
#st.title('XGBRegressor model')
st.markdown("""<h1 style='color:black'>XGBRegressor model</h1>""", unsafe_allow_html=True)
df = pd.read_csv('my_modified_data.csv')
df = pd.get_dummies(df, columns=['Date', 'Symbol'])

if 'Weighted_Price' not in df.columns:
    raise KeyError("The 'Weighted_Price' column does not exist in the DataFrame")
import os

if not os.path.exists('model.pkl'):
    raise FileNotFoundError("The 'model.pkl' file does not exist in the directory.")

df = df.reset_index(drop=True)

# Split the data into training and testing sets
X = df.drop('Weighted_Price', axis=1)
y = df['Weighted_Price']
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model =  xgb.XGBRegressor(objective ='reg:linear',min_child_weight=10, booster='gbtree', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 100)
model.fit(X_trn, y_trn,
        eval_set=[(X_trn, y_trn), (X_tst, y_tst)],
        early_stopping_rounds=50,
       verbose=False)


# Load the model from the saved file
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Create a Streamlit app to make predictions
#t.title('Bitcoin Price Prediction App')
st.markdown("""<h1 style='color:black'>Bitcoin Price Prediction App</h1>""", unsafe_allow_html=True)
st.write('Enter the features to make a prediction:')

feature_names = X.columns
features = []
for feature in feature_names:
    feature_val = st.number_input(feature, value=0.0)
    features.append(feature_val)

if st.button('Make a prediction'):
    input_data = pd.DataFrame([features], columns=feature_names)
    prediction = loaded_model.predict(input_data)
    st.write('The predicted weighted price is:', prediction[0])
    
    

# ... load and prepare data ...



# ... load and prepare data ...

# Create plot
fig, ax = plt.subplots(figsize=(15, 5))
df[['Weighted_Price','Weighted_Price_Prediction']].plot(ax=ax)

# Set plot title and axes labels
ax.set_title('Weighted Price vs Weighted Price Prediction')
ax.set_xlabel('Date')
ax.set_ylabel('Price')

# Display plot in Streamlit app
st.pyplot(fig)
