import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Set the title of the web app
st.title('Ultimate FinTech Dashboard')

# Add a text input box for the user to enter a stock ticker
ticker_symbol = st.text_input('Enter Stock Ticker', 'AAPL')

# Fetch data for the ticker
if ticker_symbol:
    # Get data on this ticker
    ticker_data = yf.Ticker(ticker_symbol)
    
    # Get the historical prices for this ticker
    ticker_df = ticker_data.history(period='1y')
    
    # --- Machine Learning Section ---
    if not ticker_df.empty:
        # Prepare Data for ML
        df_ml = ticker_df[['Close']].copy()
        df_ml['Time'] = np.arange(len(df_ml.index))
        X = np.array(df_ml['Time']).reshape(-1, 1)
        y = np.array(df_ml['Close'])

        # Train Model and Predict
        model = LinearRegression()
        model.fit(X, y)
        prediction = model.predict([[len(df_ml)]])[0]
        
        # Generate predictions for the historical data to plot the regression line
        ticker_df['Predicted Close'] = model.predict(X)

        # --- Display UI ---
        # Display the prediction
        st.subheader('Machine Learning Prediction')
        st.metric("Predicted Next Day Close", f"${prediction:.2f}")

        # Display the line chart with actual and predicted prices
        st.subheader('Closing Price Chart with Regression Line')
        st.line_chart(ticker_df[['Close', 'Predicted Close']])

        # Display the summary data table
        st.subheader('Historical Data')
        st.dataframe(ticker_df)
    else:
        st.warning('No data found for this ticker.')