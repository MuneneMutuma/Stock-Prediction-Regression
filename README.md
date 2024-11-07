# Stock Price Prediction - Regression Model

## Project Overview
This project demonstrates a **linear regression** model for predicting stock prices. The dataset used contains historical stock data for various companies from the Nairobi Stock Exchange (NSE), including key metrics such as **Day Low**, **Day High**, and **Day Price**. The goal is to predict the **Day Price** of a selected stock based on other features using regression models. The model also explores more advanced methods like **Ridge** and **Lasso** regression for comparison.

## Features

The project is broken down into the following key stages:

1. **Data Loading**: The dataset is downloaded and loaded into a pandas dataframe.
2. **Feature Selection**: The project focuses on a selected stock ('SCBK') and uses specific features for prediction, including `12m High`, `12m Low`, `Day Low`, and `Day High`.
3. **Data Preprocessing**: 
   - The date values are converted to `datetime` format for analysis.
   - Missing values are checked and handled accordingly.
   - The data is visualized for insights into the trends of the stock's performance.
4. **Feature Scaling**: 
   - The features and target variables are scaled using `StandardScaler` to standardize them before training the model.
5. **Model Training**: 
   - A **Linear Regression** model is trained using the scaled features and evaluated on the test set.
   - Advanced models such as **Ridge** and **Lasso** are also used to improve predictions by handling multicollinearity.
6. **Evaluation**: The models' performance is evaluated using **Mean Squared Error (MSE)** and **R-squared (R²)**.
7. **Visualization**: 
   - A comparison between the predicted values from Linear, Ridge, and Lasso regressions and the actual stock prices is plotted.
   - The results are visualized using **matplotlib** for static plots and **Plotly Express** for interactive graphs.

## Setup

#### Prerequisites
To run this project, ensure you have the following Python libraries installed:

- `Python 3.x`
- `pandas`
- `matplotlib`
- `numpy`
- `scikit-learn`
- `plotly`

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/MuneneMutuma/Stock-Prediction-Regression.git
   cd Stock-Prediction-Regression
   ```
2. **Install Dependeinces**
   ```bash
   pip install numpy pandas matplotlib scikit-learn plotly
   ```
3. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
4. **Run the Cells**
   Open the notebook `Regression_Stock.ipynb` and run the cells sequentially.

## Usage
1. **Data Exploration and Visualization**
   The notebook proivdes initial data exploration with visualizations of the selected stocks' performance over tim. For instance, the stock's Day Low, Day High, and Day Price are plotted to obsever trends.

2. **Model Training**
   The notebook performs the following steps for model training:
     - The data is split into training and testing datasets using `train_test_split()`
     - The features are standardized using  `StandardScaler` to normalize the data.
     - A Linear Regression model is trained, and the results are evaluated using Mean Squared Error (MSE) and R-Squared (R^2) scores.

3. **Advanced Models (Ridge & Lasso)**
   - Ridge Regression and Lasso Regression are used to see how they perform compared to liner regression. These models handle multicollinearity and perform feature selection, respectively.
   - Grid Search is used to fine-tune hyperparameters such as `alpha` for Ridge and Lasso

4. Comparison and Visualization:
   - A comparison of actual vs predicted values (using Linear, Ridge, and Lasso regression) is plotted for both training and test data.
   - The predictions from all models are displayed using matplotlib for static plots and Plotly Express for interactive plots.


## Model Evaluation
- **Mean Squared Error (MSE):** Measures the average squared difference between predicted and actual values. The lower the MSE, the better the model's performance. 
- **R-squared (R²):** Indicates how well the independent variables eplain the variation in the dependent variable. The closer to 1, the better the model.

  Here are the scores for the three models:

  
| Model               | Mean Squared Error (MSE) | R² Score  |
|---------------------|--------------------------|-----------|
| Linear Regression   | 0.049849                 | 0.978206  |
| Ridge               | 0.049593                 | 0.978318  |
| Lasso               | 0.021352                 | 0.990665  |


The best model is Lasso for both MSE and R² score
