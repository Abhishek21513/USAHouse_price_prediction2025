import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="US House Price Prediction", layout="wide")

st.title("US House Price Prediction Dashboard")
st.write("Enter house attributes to predict the price.")

# Load pre-trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Example house features
bedrooms = st.slider('Bedrooms', 1, 10, 3)
bathrooms = st.slider('Bathrooms', 1, 10, 2)
sqft_living = st.number_input('Living Area (sqft)', min_value=500, max_value=10000, value=2000)
location = st.selectbox('Location', ['New York', 'San Francisco', 'Los Angeles', 'Chicago', 'Houston'])

input_data = pd.DataFrame({
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'sqft_living': [sqft_living],
    'location': [location]
})

if st.button('Predict Price'):
    prediction = model.predict(input_data)
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")

# Optionally, add visualizations for data trends
data = pd.read_csv('data.csv')
st.subheader("Historical Price Trends")
st.line_chart(data[['date', 'price']].set_index('date'))

