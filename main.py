from src.model import LinearRegression
from sklearn.linear_model import LinearRegression as SkLearnLinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import RAW_DATA_DIR
from src.utils import load_and_preprocess_data
from src.visualization import show
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

# The class StandardScaler is used to standardize features by removing the mean and scaling to unit variance.
scaler = StandardScaler()
# Importing the data using Pandas.
df = load_and_preprocess_data(f'{RAW_DATA_DIR}/train.csv')

X = df[['TotalLivingArea']].values
y = df['SalePrice'].to_numpy()

scaler.fit(X)
# Scaling the features
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training our custom model.
# .flatten() is used to convert the 2D array into a 1D array.
model = LinearRegression(X_train.flatten(), y_train)
sk_model = SkLearnLinearRegression()
sk_model.fit(X_train, y_train)

# sk_model.coef_[0] is the slope of SkLearn Linear Regression, then it is passed to float and rounded.
logger.info(f"\n\nSlope (Inclinação)\nCustom Model| {round(model.calculate_slope(), 2)}\nSkLearn      | {round(float(sk_model.coef_[0]), 2)}")
logger.info(f"\n\nIntercept (Interceção)\nCustom Model | {round(model.calculate_intercept(), 2)}\nSkLearn      | {round(float(sk_model.intercept_), 2)}")
logger.info(f"\n\nR² (Determination Coefficient)\nCustom Model | {round(model.calculate_rsquared(), 2)}\nSkLearn      | {round(float(sk_model.score(X, y)), 2)}")
# X.flatten() is used to convert the 2D array into a 1D array.
# It is used [0][1] because the return value of np.corrcoef is a 2D array.
logger.info(f"\n\nCorrelation\nCustom Model | {round(model.calculate_correlation(), 2)}\nSkLearn      | {round(np.corrcoef(X.flatten(), y.flatten())[0][1], 2)}")

y_pred_test = model.predict(X_test.flatten())

show(model, highlight=(X_test, y_pred_test))