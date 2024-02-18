import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
import numpy as np

# Function to fetch historical stock data within a specified time frame
def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1wk')
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Function to robustly normalize data
def robust_normalize(series):
    # Drop NaN values before normalization
    series_cleaned = series.dropna()

    if series_cleaned.empty:
        # Return an empty series if all values are NaN
        return pd.Series(index=series.index)

    return (series_cleaned - series_cleaned.median()) / (series_cleaned.quantile(0.75) - series_cleaned.quantile(0.25))

# Function to calculate dynamic time warping distance
def calculate_dtw_distance(series1, series2):
    return np.sum(np.abs(series1 - series2))

# Define the time frame (8 weeks)
end_date = pd.to_datetime('2024-02-15')
start_date = end_date - pd.DateOffset(weeks=8)

# Fetch historical data for Apple (AAPL), Microsoft (MSFT), Google (GOOGL), and Amazon (AMZN)
stocks_to_compare = ['IREDA.NS', 'RVNL.NS', 'ITC.NS', 'RELIANCE.NS']
stock_data_compare = {}

for stock_ticker in stocks_to_compare:
    stock_data = fetch_stock_data(stock_ticker, start_date, end_date)
    if stock_data is not None:
        stock_data_compare[stock_ticker] = stock_data
        stock_data_compare[stock_ticker]['Normalized_Close'] = robust_normalize(stock_data['Close'])

# Fetch historical data for the stock to be compared (replace 'FB' with 'NVDA' for example)
compare_stock_ticker = 'VBL.NS'
stock_data_compare_stock = fetch_stock_data(compare_stock_ticker, start_date, end_date)

if stock_data_compare_stock is not None:
    normalized_compare_stock = robust_normalize(stock_data_compare_stock['Close'])

    # Reshape data for tslearn
    X = []
    for stock_ticker, stock_data in stock_data_compare.items():
        normalized_stock = stock_data['Normalized_Close']
        X.append(normalized_stock.values.reshape(-1, 1))
    X = np.array(X)

    # Reshape the stock to be compared
    X_compare = normalized_compare_stock.values.reshape(1, -1, 1)  # Reshape to (1, sequence_length, n_features)

    # Scale the data
    X = TimeSeriesScalerMinMax().fit_transform(X)
    X_compare = TimeSeriesScalerMinMax().fit_transform(X_compare)

    # Fit Time Series KMeans model
    n_clusters = len(stocks_to_compare)
    model = TimeSeriesKMeans(n_clusters=n_clusters, verbose=True)
    model.fit(X)

    # Predict the cluster of the stock to be compared
    cluster_prediction = model.predict(X_compare)

    # Identify the most similar stock based on cluster
    most_similar_stock_idx = np.argmin(np.sum(np.abs(model.cluster_centers_ - X_compare), axis=(1, 2)))
    most_similar_stock = stocks_to_compare[most_similar_stock_idx]

    print(f"The most similar stock to {compare_stock_ticker} is {most_similar_stock}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(stock_data_compare_stock.index, normalized_compare_stock, label=compare_stock_ticker)
    plt.plot(stock_data_compare[most_similar_stock].index, stock_data_compare[most_similar_stock]['Normalized_Close'], label=most_similar_stock)
    plt.xlabel('Date')
    plt.ylabel('Robust Normalized Closing Price')
    plt.title(f'Normalized Stock Price Comparison ({compare_stock_ticker} vs. {most_similar_stock})')
    plt.legend()

    # Plotting Dissimilar Graphs
    plt.subplot(2, 1, 2)
    for stock_ticker in stocks_to_compare:
        if stock_ticker != most_similar_stock:
            plt.plot(stock_data_compare[stock_ticker].index, stock_data_compare[stock_ticker]['Normalized_Close'], label=stock_ticker)

    plt.xlabel('Date')
    plt.ylabel('Robust Normalized Closing Price')
    plt.title('Dissimilar Stock Price Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()
