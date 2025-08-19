# Project: Linear Regression from Scratch

## Description
This project implements a Simple Linear Regression model in pure Python (using only NumPy) to predict house prices based on their total area. The goal is to demonstrate understanding of the calculations behind the model, comparing the results with the `scikit-learn` library for validation.

## How to Run

1.  **Clone the Repo**
    ```bash
    git clone [https://github.com/VictorHFerreira016/linear-regression-from-scratch.git](https://github.com/VictorHFerreira016/linear-regression-from-scratch.git)
    ```

2.  **Install Dependecies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Main**
    ```bash
    python src/main.py
    ```
    The program will train the models, display the comparisons, and then ask you to enter an area (in square feet) to predict the property price.

## Project Structure
```
.
├── data/
│   └── train.csv         # Train data
├── notebooks/
│   └── Exploration.ipynb # Notebook for analisys
├── src/
│   ├── main.py           # Main script
│   ├── model.py          # Class of the model
│   └── visualization.py  # To generate the graph
├── .gitignore            # Files to ignore
└── README.md             # This file
```

## Example

The script generates a direct comparison between the coefficients of the manually created model and the `scikit-learn` library, as well as a scatterplot with the regression line.
