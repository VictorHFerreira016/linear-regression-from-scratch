import matplotlib.pyplot as plt
from model import LinearRegression

def show(model: LinearRegression, highlight=None):
    '''Visualize the linear regression model.
    args:
        model: The linear regression model to visualize.
        highlight: Optional tuple (x, y) to highlight a specific point.
    '''
    
    plt.scatter(model.features, model.targets, color='blue', label='Real Data')

    y_pred_line = model.calculate_intercept() + model.calculate_slope() * model.features
    plt.plot(model.features, y_pred_line, color='red', label='Linear Regression')

    # If highlight argument is provided, plot the point
    if highlight:
        # Unpack the highlight tuple
        x_highlight, y_highlight = highlight
        # s: size of the highlight point
        plt.scatter(x_highlight, y_highlight, color='red', s=100, marker='X', label=f'Prediction: {y_highlight:.2f}')

    plt.xlabel('Features')
    plt.ylabel('Targets')
    plt.title('Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'images/{x_highlight}_linear_regression_plot.png', dpi=300)
    plt.show()