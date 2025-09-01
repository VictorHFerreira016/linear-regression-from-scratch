import matplotlib.pyplot as plt
from src.model import LinearRegression 
import logging
from src.config import IMAGES_DIR

logger = logging.getLogger(__name__)

def show(model: LinearRegression, highlight: tuple | None = None):
    """
    Visualize the linear regression model.
    args:
        model: The linear regression model to be visualized.
        highlight: Optional tuple (x, y) to highlight a specific point.
    """
    
    # Check if the model has been trained
    if model.features is None or model.targets is None:
        logger.warning("Model not trained or no data available. Cannot generate visualization.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(model.features, model.targets, color='blue', alpha=0.6, label='Real Data')
    
    y_pred_line = model.calculate_intercept() + model.calculate_slope() * model.features
    plt.plot(model.features, y_pred_line, color='red', linewidth=2, label='Regression Line')

    if highlight:
        x_highlight, y_highlight = highlight

        plt.scatter(x_highlight, y_highlight, color='green', s=150, marker='x', alpha=0.8, label="Test Point ")

        plt.savefig(f"{IMAGES_DIR}/highlighted_regression_plot.png", dpi=300)
    else:
        plt.savefig(f'{IMAGES_DIR}/own_linear_regression_plot.png', dpi=300)

    plt.xlabel('Total Area (Standardized)')
    plt.ylabel('Targets')
    plt.title('Linear Regression Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()