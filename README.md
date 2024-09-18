# Bitcoin Price Prediction Model 💰

Welcome to the **Bitcoin Price Prediction Model**! This repository contains a deep learning-based Streamlit application designed to predict future Bitcoin prices based on historical data. The model uses past price data to project future values and allows users to visualize both recent price trends and predicted future prices.

## Features

- 📈 **Recent Bitcoin Price Data**: Display recent historical Bitcoin price data from Yahoo Finance.
- 🔮 **Future Price Predictions**: Predict Bitcoin prices for up to 30 days into the future using a pre-trained deep learning model.
- 📊 **Real-time Visualization**: View both recent and predicted prices in a clear and intuitive line chart.
- 🖥️ **Customizable Dashboard**: Users can select the number of recent prices to display and the number of future days to predict.
- 💡 **Streamlit Interface**: Easy-to-use web-based interface built with Streamlit for real-time interaction.

## How to Use

1. **Run the Streamlit App**: Use the link below to access the live app.
   - [Live App](https://cyberfantics-bitcoin-price-prediction-app-cwwesw.streamlit.app/)

2. **Clone the Repository**: If you prefer to run the app locally, clone the repository to your system:
   ```bash
   git clone https://github.com/cyberfantics/bitcoin-price-prediction.git
   ```
3. **Install Dependencies:** Install the required Python libraries:
   ```pip install -r requirements.txt```
4. **Run the App Locally:** Navigate to the project directory and run the app using Streamlit:
   ```streamlit run app.py```
5. **Explore the Predictions:** Adjust the sidebar options to select the number of recent price records to display and the number of future days to predict.

## Model Information
- The model used for predictions is a deep learning model trained on historical Bitcoin prices using a LSTM (Long Short-Term Memory) network.
- It leverages Yahoo Finance data to download Bitcoin's historical prices.
- The data is preprocessed using MinMaxScaler to normalize the input for the neural network.

## Code Overview
- **app.py:** The main Streamlit application that handles the user interface, data fetching, and model prediction logic.
- **model/Bitcoin_Future_Price_prediction_Model.keras:** Pre-trained deep learning model for Bitcoin price prediction.
- **requirements.txt:** Python dependencies needed to run the app.

## Disclaimer
This model is for educational purposes only and should not be used as financial advice. Always perform your own research before making investment decisions.

#### Built With ❤️ by [Syed Mansoor ul Hassan Bukhari](https://github.com/cyberfantics)
