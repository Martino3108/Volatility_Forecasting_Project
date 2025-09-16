## Volatility Forecasting Project (Using the Optiver Realized Volatility Challenge Dataset)

This project aims to predict the realized volatility of financial assets using various forecasting techniques, leveraging high-frequency trading data. The dataset consists of order book and trade data for more than 100 stocks, organized in 10 minutes time windows. The project involves the use of both statistical and deep learning approaches for time series prediction.


## Methodology

### 1. **More Statistical Approach**:
   - I extracted punctual features from order book and trade data for each time window and stock, including past volatility, kurtosis of returns, and GJR-based volatility forecasts.
   - linear regression models are then trained on these features to predict future realized volatility.
   
### 2. **Deep Learning Approach (LSTM)**:
   - The LSTM model is fed with multivariate time series data, capturing the dynamics of order book features and traded volumes.
   - This approach leverages sequential data, learning directly from the time series without requiring handcrafted summary statistics.


### Notebooks in the Project

1. **[GJR_forecasting_project.ipynb](./GJR_forecasting_project.ipynb)**: 
   - This notebook contains the computation of the conditional volatility of a GJR model for each time window.

2. **[volatility_data_forecasting.ipynb](./volatility_data_forecasting.ipynb)**: 
   - This notebook compute punctual volatility based features for each time window.

3. **[LSTM_forecasting_project.ipynb](./LSTM_forecasting_project.ipynb)**: 
   - This file implements the deep learning approach using Long Short-Term Memory (LSTM) networks to predict realized volatility. 

4. **[Forecasting_Project.R](./Forecasting_Project.R)**: 
   - The model implemented are standard linear regressions and linear regressions with logaritmic transformation of the variabls. Fixed effect based on the stock is used.
   - The R script focuses on cleaning the data, testing different specifications of volatility (standard and weighted), assessing residuals for homoscedasticity and normality, and selecting the best-fitting model based on statistical tests and performance (RMSPE).


## Evaluation Metrics

- The **Root Mean Squared Percentage Error (RMSPE)** is used to evaluate the performance of the models, as suggested in the original challenge.

## Dataset

Due to the size of the dataset, it cannot be uploaded here. However, you can find the dataset on Kaggle:
- [Optiver Realized Volatility Prediction Dataset](https://www.kaggle.com/c/optiver-realized-volatility-prediction/data)

### Data Description
The dataset consists of:
- **book_train.parquet**: Order book snapshots (bid and ask prices/sizes).
- **trade_train.parquet**: Recorded trades during each 10-minute interval.
- **train.csv**: Realized volatility for each stock over a 10-minute window.

