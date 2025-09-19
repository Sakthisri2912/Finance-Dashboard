import streamlit as st
import yfinance as yf

st.title('Ultimate FinTech Dashboard')

# Add a text input box for the user to enter a stock ticker
ticker_symbol = st.text_input('Enter Stock Ticker', 'AAPL')

# Fetch data for the ticker
if ticker_symbol:
    # Get data on this ticker
    ticker_data = yf.Ticker(ticker_symbol)
    
    # Get the historical prices for this ticker
    ticker_df = ticker_data.history(period='1y')
    
    # Display the line chart
    st.subheader('Closing Price Chart')
    st.line_chart(ticker_df.Close)

    # Display the summary data table
    st.subheader('Historical Data')
    st.dataframe(ticker_df)