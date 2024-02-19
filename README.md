Stock Price Comparison using Time Series Clustering
Overview
This project aims to provide a comprehensive comparison of historical stock prices for different companies using Time Series KMeans clustering. By applying clustering techniques to time series data, we can identify stocks with similar price movements over a specified time frame.

Features
Data Retrieval: The project leverages the Yahoo Finance API to fetch historical stock price data for a set of companies.

Time Series Clustering: Utilizing the tslearn library, the script applies Time Series KMeans clustering to group stocks based on the similarity of their price movements.

Similar Stock Identification: Given a target stock, the script identifies the most similar stock within the clustering results.

Visualization: The project provides visualizations of normalized stock prices for easy comparison.

Requirements
Python 3.x
Libraries:
yfinance
matplotlib
pandas
tslearn
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/KadiyaheetChetanbhai/Stock_Annalyzing_Python.git
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Configuration:

Modify the stocks_to_compare list in the stock_comparison.py script to include the stock tickers you want to compare.
Run the script:

bash
Copy code
python stock_comparison.py
View Results:

The script will fetch historical stock data, perform clustering, and visualize the comparison results.
Contributors
Your Name
Heet kadiya


Acknowledgements
Yahoo Finance API
tslearn library
Contact
For questions or suggestions, please feel free to contact the project maintainers:

Your Name: Heet kadiya
Contributor 1: heetkadiya04@gmail.com
