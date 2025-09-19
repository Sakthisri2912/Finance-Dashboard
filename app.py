import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Set the title of the web app
st.title('Ultimate FinTech Dashboard')

# Add a text input box for the user to enter a stock ticker
ticker_symbol = st.text_input('Enter Stock Ticker', 'AAPL')

# Fetch data for the ticker
if ticker_symbol:
    # Get data on this ticker
    ticker_data = yf.Ticker(ticker_symbol)
    
    # Get the historical prices for this ticker
    ticker_df = ticker_data.history(period='2y') # Increased period for better feature calculation
    
    # --- Advanced Feature Engineering ---
    if not ticker_df.empty:
        df_ml = ticker_df[['Close']].copy()
        
        # Create Moving Averages
        df_ml['MA7'] = df_ml['Close'].rolling(window=7).mean()
        df_ml['MA21'] = df_ml['Close'].rolling(window=21).mean()
        
        # Create Lag Feature
        df_ml['Lag1'] = df_ml['Close'].shift(1)
        
        # Remove rows with NaN values created by feature engineering
        df_ml.dropna(inplace=True)

        # --- Prepare Data for ML ---
        # Features (X) are the engineered features
        X = df_ml[['MA7', 'MA21', 'Lag1']]
        # Target (y) is the 'Close' price
        y = df_ml['Close']
        
        # --- Train Model and Predict ---
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Prepare the last row of data to predict the next day
        last_row = X.iloc[[-1]]
        prediction = model.predict(last_row)[0]
        
        # --- THE FIX IS HERE ---
        # Generate predictions for the historical data to plot the model fit
        # We align the predictions with the original dataframe's index
        predictions = model.predict(X)
        ticker_df['Predicted Close'] = pd.Series(predictions, index=X.index)

        # --- Display UI ---
        # Display the prediction
        st.subheader('Advanced ML Prediction (Random Forest + Features)')
        st.metric("Predicted Next Day Close", f"${prediction:.2f}")

        # Display the line chart with actual and predicted prices
        st.subheader('Closing Price Chart with Model Fit')
        st.line_chart(ticker_df[['Close', 'Predicted Close']])

        # Display the summary data table
        st.subheader('Historical Data')
        st.dataframe(ticker_df)
    else:
        st.warning('No data found for this ticker.')