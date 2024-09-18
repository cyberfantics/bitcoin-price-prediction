import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import os

# Load the pre-trained model from the 'model' folder
model_path = os.path.join('model', 'Bitcoin_Future_Price_prediction_Model.keras')
model = load_model(model_path)

# Set Streamlit page configuration with wide layout
st.set_page_config(
    page_title='Bitcoin Price Prediction', 
    page_icon='💰', 
    layout='wide', 
    initial_sidebar_state='expanded'
)

# Sidebar information
st.sidebar.title("📊 Bitcoin Price Prediction Dashboard")
st.sidebar.info("Built with ❤️ by **Syed Mansoor ul Hassan Bukhari**")
st.sidebar.write("[GitHub Repository](https://github.com/cyberfantics/bitcoin-price-prediction)")

# Sidebar option to select number of recent price records to show
st.sidebar.markdown("<p style='font-size:20px;'>📈 Select number of recent Bitcoin prices to display:</p>", unsafe_allow_html=True)
recent_price_records = st.sidebar.number_input('', min_value=1, max_value=20, value=5)  # Default is 5 records

# Sidebar option to select number of future days to predict
st.sidebar.markdown("<p style='font-size:20px;'>⏳ Select number of future days to predict:</p>", unsafe_allow_html=True)
future_days = st.sidebar.number_input(' ', min_value=1, max_value=30, value=5)

# Main Title and Description
st.title('💰 Bitcoin Future Price Prediction Model')
st.markdown("""
    Welcome to the **Bitcoin Future Price Prediction Model**! 
    This tool uses a deep learning model to predict future Bitcoin prices based on historical data. 
    You can select the number of days in the future you want to predict and see the projected prices visualized in real-time.
""")

# Custom styling for text with color and font size
st.markdown("""
    <style>
    .title-font {
        font-size:25px !important;
        color:#ff6347;
        font-weight: bold;
    }
    .info-text {
        font-size:18px;
        color:#4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Download Bitcoin price data from Yahoo Finance
data = pd.DataFrame(yf.download('BTC-USD', start='2015-01-01', end='2024-09-18'))
data.reset_index(inplace=True)

# Prepare data for predictions
train_data = data[:-100]['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)

# Future price predictions
future_predictions = []

# Get the last known prices for predictions
last_prices = train_data_scaled[-100:]  # Use the last 100 prices as the starting point

for _ in range(future_days):
    # Prepare the input for the model
    input_data = last_prices[-100:].reshape(1, 100, 1)
    future_pred = model.predict(input_data)
    
    # Append prediction and update the last prices
    future_predictions.append(future_pred[0, 0])
    last_prices = np.append(last_prices, future_pred)[-100:]  # Keep only the last 100 prices

# Scale back future predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create two columns for displaying the recent and predicted prices side by side
col1, col2 = st.columns(2)

# Display recent Bitcoin prices in the first column
with col1:
    st.subheader('📈 Recent Bitcoin Price Data')
    st.write(data[['Date', 'Close']].tail(recent_price_records))  # Show selected number of recent records

# Display future price predictions in the second column
with col2:
    st.subheader('🔮 Predicted Future Bitcoin Prices')
    future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_days)
    future_prices_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Future Price': future_predictions.flatten()
    })
    st.write(future_prices_df)

# Visualizing the future prices in a line chart (below the two columns)
st.line_chart(future_prices_df.set_index('Date'))

# Adding Footer Information
st.markdown("""
    <hr>
    <p class='info-text'>
    This model is designed for educational purposes and should not be used for financial advice.
    </p>
    <p class='info-text'>
    For any feedback or contributions, visit the repository linked in the sidebar.
    </p>
""", unsafe_allow_html=True)
