import yfinance as yf

# Download stock data for a specific ticker
ticker = 'AAPL'
stock_data = yf.download(ticker, start='2020-01-01', end='2023-01-01')
print(stock_data.head())
# data storage
stock_data.to_csv('AAPL_stock_data.csv')
# data cleaning
import pandas as pd
stock_data = pd.read_csv('AAPL_stock_data.csv', index_col='Date', parse_dates=True)
stock_data.isnull().sum()  # Check for missing values
stock_data.dropna(inplace=True)  # Drop rows with missing values

# data exploration
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(stock_data['Close'])
plt.title('Stock Closing Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()

print(stock_data.describe())
# data standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(stock_data[['Close']])

# feature selection and data partitioning
from sklearn.model_selection import train_test_split

X = stock_data.index.values.reshape(-1, 1)
y = stock_data['Close'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print('KNN MSE:', mean_squared_error(y_test, y_pred_knn))

# Vector Machine (SVM) Models for Stock Quotes
from sklearn.svm import SVR

svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
print('SVR MSE:', mean_squared_error(y_test, y_pred_svr))

# Bayesian Model for Stock Quotes
from sklearn.linear_model import BayesianRidge

bayesian_ridge = BayesianRidge()
bayesian_ridge.fit(X_train, y_train)
y_pred_bayes = bayesian_ridge.predict(X_test)
print('Bayesian Ridge MSE:', mean_squared_error(y_test, y_pred_bayes))

# Random Forest Models for Stock Quotes
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print('Random Forest MSE:', mean_squared_error(y_test, y_pred_rf))

# Model Comparison through Report and ROC
from sklearn.metrics import mean_squared_error, r2_score

models = {
    'KNN': (y_test, y_pred_knn),
    'SVR': (y_test, y_pred_svr),
    'Bayesian Ridge': (y_test, y_pred_bayes),
    'Random Forest': (y_test, y_pred_rf)
}

for name, (true, pred) in models.items():
    print(f'{name} - MSE: {mean_squared_error(true, pred)}, R2: {r2_score(true, pred)}')

# Analysis of factors affecting stock marketing
# correlation analysis
correlations = stock_data.corr()
print(correlations)

# Regression analysis
from sklearn.linear_model import LinearRegression

X_factors = stock_data[['Volume']]  # Example factor
y_prices = stock_data['Close']

lin_reg = LinearRegression()
lin_reg.fit(X_factors, y_prices)
y_pred_reg = lin_reg.predict(X_factors)
print('Regression Coefficients:', lin_reg.coef_)
