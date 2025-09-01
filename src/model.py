import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class LinearRegression:
    """
    A simple linear regression implementation from scratch.
    
    This class implements ordinary least squares linear regression
    for single-variable problems using the normal equation method.
    
    Parameters
    ----------
    features : np.ndarray
        Input features (X) of shape (n_samples, 1)
    targets : np.ndarray  
        Target values (y) of shape (n_samples, 1)
        
    Attributes
    ----------
    slope_ : float
        The slope coefficient of the fitted line
    intercept_ : float
        The intercept of the fitted line
    r_squared_ : float
        The coefficient of determination
    correlation_ : float
        The Pearson correlation coefficient
    """
    
    def __init__(self, features: Optional[np.ndarray] = None, targets: Optional[np.ndarray] = None):
        self.features: Optional[np.ndarray] = None
        self.targets: Optional[np.ndarray] = None
        self.slope_: Optional[float] = None
        self.intercept_: Optional[float] = None
        self.r_squared_: Optional[float] = None
        self.correlation_: Optional[float] = None
        self._is_fitted: bool = False
        
        # Auto-fit if both features and targets are provided (backward compatibility)
        if features is not None and targets is not None:
            self.fit(features, targets)
            
    def fit(self, features: np.ndarray, targets: np.ndarray) -> 'LinearRegression':
        """
        Fit the linear regression model.
        
        Parameters
        ----------
        features : np.ndarray
            Training features of shape (n_samples, 1)
        targets : np.ndarray
            Training targets of shape (n_samples, 1)
            
        Returns
        -------
        self : LinearRegression
            Returns the fitted estimator
        """
        self.features = self._validate_input(features)
        self.targets = self._validate_input(targets)
        
        if len(self.features) != len(self.targets):
            raise ValueError("Features and targets must have the same length")
            
        self.slope_ = self._calculate_slope()
        self.intercept_ = self._calculate_intercept()
        self.correlation_ = self._calculate_correlation()
        self.r_squared_ = self.correlation_ ** 2
        self._is_fitted = True
        
        logger.info(f"Model fitted: slope={self.slope_:.4f}, intercept={self.intercept_:.4f}, R²={self.r_squared_:.4f}")
        return self
    
    def _validate_input(self, data: np.ndarray) -> np.ndarray:
        """Validate and reshape input data."""
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim == 2 and data.shape[1] != 1:
            raise ValueError("Only single-feature regression is supported")
            
        return data
    
    def _calculate_intercept(self) -> float:
        """Calculate the y-intercept using the normal equation."""
        if self.features is None or self.targets is None:
            raise ValueError("Model must be fitted first")
            
        n = len(self.features)
        x = self.features.flatten()
        y = self.targets.flatten()
        
        numerator = (np.sum(y) * np.sum(x ** 2) - np.sum(x) * np.sum(x * y))
        denominator = (n * np.sum(x ** 2) - (np.sum(x)) ** 2)
        
        if denominator == 0:
            raise ValueError("Cannot calculate intercept: denominator is zero")
            
        return float(numerator / denominator)
    
    def _calculate_slope(self) -> float:
        """Calculate the slope using the normal equation."""
        if self.features is None or self.targets is None:
            raise ValueError("Model must be fitted first")
            
        n = len(self.features)
        x = self.features.flatten()
        y = self.targets.flatten()
        
        numerator = (n * np.sum(x * y) - np.sum(x) * np.sum(y))
        denominator = (n * np.sum(x**2) - (np.sum(x))**2)
        
        if denominator == 0:
            raise ValueError("Cannot calculate slope: denominator is zero")
            
        return float(numerator / denominator)

    def _calculate_correlation(self) -> float:
        """Calculate the Pearson correlation coefficient."""
        if self.features is None or self.targets is None:
            raise ValueError("Model must be fitted first")
            
        x = self.features.flatten()
        y = self.targets.flatten()
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        x_var = np.sum((x - x_mean)**2)
        y_var = np.sum((y - y_mean)**2)
        
        if x_var == 0 or y_var == 0:
            return 0.0
            
        denominator = np.sqrt(x_var * y_var)
        return float(numerator / denominator)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Parameters
        ----------
        X : np.ndarray
            Input features for prediction
            
        Returns
        -------
        y_pred : np.ndarray
            Predicted values
        """
        if not self._is_fitted or self.slope_ is None or self.intercept_ is None:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
            
        X = self._validate_input(X)
        return self.intercept_ + self.slope_ * X.flatten()
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the R² score for the model.
        
        Parameters
        ----------
        X : np.ndarray
            Test features
        y : np.ndarray
            True target values
            
        Returns
        -------
        score : float
            R² score
        """
        if not self._is_fitted or self.slope_ is None or self.intercept_ is None:
            raise ValueError("Model must be fitted before calculating score. Call fit() first.")
            
        y_pred = self.predict(X)
        y_true = self._validate_input(y).flatten()
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
            
        return float(1 - (ss_res / ss_tot))
    
    def get_params(self) -> dict:
        """Get model parameters."""
        if not self._is_fitted or self.slope_ is None or self.intercept_ is None or self.r_squared_ is None or self.correlation_ is None:
            return {}
        return {
            'slope': self.slope_,
            'intercept': self.intercept_,
            'r_squared': self.r_squared_,
            'correlation': self.correlation_
        }
    
    # Backward compatibility methods
    def calculate_slope(self) -> float:
        """Backward compatibility method."""
        if not self._is_fitted or self.slope_ is None:
            raise ValueError("Model must be fitted first")
        return self.slope_
    
    def calculate_intercept(self) -> float:
        """Backward compatibility method."""
        if not self._is_fitted or self.intercept_ is None:
            raise ValueError("Model must be fitted first")
        return self.intercept_
    
    def calculate_correlation(self) -> float:
        """Backward compatibility method."""
        if not self._is_fitted or self.correlation_ is None:
            raise ValueError("Model must be fitted first")
        return self.correlation_
    
    def calculate_rsquared(self) -> float:
        """Backward compatibility method."""
        if not self._is_fitted or self.r_squared_ is None:
            raise ValueError("Model must be fitted first")
        return self.r_squared_
    
    def __repr__(self) -> str:
        if self._is_fitted and self.slope_ is not None and self.intercept_ is not None:
            return f"LinearRegression(slope={self.slope_:.4f}, intercept={self.intercept_:.4f})"
        return "LinearRegression(not fitted)"