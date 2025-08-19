import numpy as np

class LinearRegression:
    '''Class to Create a Linear Regression Instance'''
    # First we get the dependent and independent variables.
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = features
        self.targets = targets

    def calculate_intercept(self) -> float:
        '''Function to calculate the intercept.'''
        # n is the number of data points
        n = len(self.features)
        # x is the feature values.
        x = self.features
        # y is the target values, the variable that we want to predict.
        y = self.targets
        # a is the intercept, it is a function of the means and slopes.
        a = (np.sum(y) * np.sum(x ** 2) - np.sum(x) * np.sum(x * y)) / (n * np.sum(x ** 2) - (np.sum(x)) ** 2)
        return a
    
    def calculate_slope(self) -> float:
        '''Function to calculate the slope'''
        # n is the number of data points
        n = len(self.features)
        # x is the feature values.        
        x = self.features
        # y is the target values, the variable that we want to predict.
        y = self.targets
        # m is the slope
        m = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
        return m

    def calculate_correlation(self) -> float:
        '''Function to calculate the correlation coefficient.'''
        # This function calculates the correlation coefficient.
        x = self.features.flatten()
        y = self.targets.flatten()
        # The function of the correlation.
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))

        r = numerator / denominator
        
        return float(r)
    
    def calculate_rsquared(self) -> float:
        '''Function to calculate R² to measure the goodness of fit.'''
        rSquared = self.calculate_correlation()**2
        return float(rSquared)

    def predict(self, x: float, show_point: bool = False) -> float:
        '''Function to predict the value of y based on the best-fit line.'''
        y = self.calculate_intercept() + (self.calculate_slope()*x)
        if show_point:
            from visualization import show
            show(self, highlight=(x, y))
        print(y)
        return y