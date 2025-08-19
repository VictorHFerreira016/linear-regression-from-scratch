import pandas as pd
from model import LinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
import numpy as np
from sklearn.preprocessing import StandardScaler

# The class StandardScaler is used to standardize features by removing the mean and scaling to unit variance.
scaler = StandardScaler()
# Importing the data using Pandas.
# This is a DataSet of House Prices, based on various factors. 
df = pd.read_csv('data/train.csv')
print(df.loc[:, ['1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFinSF1', 'BsmtFinSF2', 'SalePrice']].head())

# But we need just the TotalLivingArea, combining all the areas of the house.
df['TotalLivingArea'] = (
    df['1stFlrSF'] + df['2ndFlrSF'] +
    df['LowQualFinSF'] + df['GrLivArea'] +
    df['BsmtFinSF1'] + df['BsmtFinSF2']
)

#X_scaled = scaler.fit_transform(df['TotalLivingArea'].values.reshape(-1, 1))
X = df['TotalLivingArea'].values.reshape(-1, 1)
y = df['SalePrice'].values.reshape(-1, 1)

model = LinearRegression(X, y)
sk_model = SklearnLinearRegression()
sk_model.fit(X, y)

print(f"\n\nSlope\nOwn Linear Regression | {round(model.calculate_slope(), 2)}\nSkLearn Linear Regression | {round(sk_model.coef_[0][0], 2)}")
print(f"\n\nIntercept\nOwn Linear Regression | {round(model.calculate_intercept(), 2)}\nSkLearn Linear Regression | {round(sk_model.intercept_[0], 2)}")
print(f"\n\nR²\nOwn Linear Regression | {round(model.calculate_rsquared(), 2)}\nSkLearn Linear Regression | {round(sk_model.score(X, y), 2)}")
print(f"\n\nCorrelation\nOwn Linear Regression | {round(model.calculate_correlation(), 2)}\nSkLearn Linear Regression | {round(np.corrcoef(X.flatten(), y.flatten())[0][1], 2)}")

X_test = float(input("\nEnter a Total Living Area to predict the House Price: "))
print(f"\n\nSkLearn Linear Regression: {sk_model.predict([[X_test]])}")
print(f"Own Linear Regression: {model.predict(X_test, show_point=True)}")