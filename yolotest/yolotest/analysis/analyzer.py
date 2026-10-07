import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import SplineTransformer
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# data cleaning split dataset
def data_clean(data_directory):
    # create pandas dataframe
    data = []
    # read all datasets
    for filename in os.listdir(data_directory):
        filepath = os.path.join(data_directory, filename)
        temp = pd.read_csv(filepath)
        name = temp["filename"]
        # extract fov
        temp["fov"] = name.str.split("--", n=1).str[0].astype(float)
        # extract residuals
        temp["residuals"] = temp["true_distance"] - temp["prediction"]
        data.append(temp)
    data = pd.concat(data, ignore_index=True)
    # drop rows that are na for prediction
    cleaned_data = data[data["prediction"] != -1]
    # split the dataset
    x = cleaned_data[["prediction", "fov"]]
    y = cleaned_data[["residuals", "true_distance"]]
    return x,y
    
# function to plot our data
def res_scatter_plot_3d(fov, predicted, residual):
    plot = plt.figure()
    axes = plot.add_subplot(projection="3d")
    axes.scatter(fov, predicted, residual, c=residual, cmap="viridis")
    axes.set_xlabel("fov")
    axes.set_ylabel("predicted")
    axes.set_zlabel("residual")
    return plot

# function to plot predctied residuals - actual residuals error against explanatory variables to ensure there is roughly random variance
def res_scatter_plot_2d(residuals, predicted_residuals, ind_var):
    calibration_error = residuals-predicted_residuals
    plot = plt.figure()
    axes = plot.add_subplot()
    axes.scatter(ind_var, calibration_error)
    axes.set_xlabel(ind_var.name)
    axes.set_ylabel("calibration_error")
    return plot

# function to anlayze our model (lin reg)
def model_analysis(xtr, ytr):
    y = ytr.drop(columns=["true_distance"]).to_numpy().flatten()
    xtr['prediction_squared'] = xtr['prediction'] ** 2
    X = sm.add_constant(xtr)
    standard_model = sm.OLS(y, X).fit()
    return standard_model

# evaluate a model on perfromance with cross val
# specifically used for residual models to reincorporate the raw model prediction then get MAE
def evaluate_residual_model(model, x, y, cv=5):
    # create folds of data
    kfold = KFold(
        n_splits=cv,
        shuffle=True,
    )
    # predict residuals across folds
    predicted_residuals = cross_val_predict(
        model,
        x,
        y["residuals"],
        cv=kfold
    )
    # map predicted residuals with original predictions for calibrated true distance prediction
    corrected_predictions = (
        x["prediction"] +
        predicted_residuals
    )
    # calculate mae
    mae = abs(y["true_distance"] - corrected_predictions).mean()

    return mae

# evaluate a model on perfromance with cross val
# specifically used for true distance models
def evaluate_true_distance_model(model, x, y, cv=5):
    # cross validate 
    mae = -cross_val_score(
        model,
        x,
        y["true_distance"],
        cv=cv,
        scoring="neg_mean_absolute_error"
    ).mean()

    return mae

# perform grid search for hyperparameters
def hyperparam_study_splines(model, x, y):
    # parameter search space for splines
    param_grid = {"splinetransformer__n_knots": [3, 5, 7, 9], "splinetransformer__degree": [1,2,3]}
    # perform search
    search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="neg_mean_absolute_error")
    search.fit(x, y['residuals'])
    # return best model and mae
    best_model = search.best_estimator_
    best_mae = -search.best_score_
    return best_model, best_mae

# helper function to save models
def save_model(model, model_name):
    model_dir = Path(__file__).resolve().parent / "models"
    joblib.dump(model, model_dir / f"{model_name}.joblib")  

if __name__ == "__main__":
    # clean and split data
    data_dir = Path(__file__).resolve().parent.parent / "datasets" / "logs" 
    x, y = data_clean(data_dir)
    xtr, xtst, ytr, ytst = train_test_split(x,y, shuffle=True, test_size=0.2)
    # plot data 
    fig = res_scatter_plot_3d(x['fov'], x['prediction'], y['residuals'])
    plt.show(block=True)
    # Initialize Models
    gb_model = GradientBoostingRegressor(
        n_estimators = 100,
        learning_rate = 0.05,
        max_depth = 3,
        )
    gb_comp = GradientBoostingRegressor(
        n_estimators = 100,
        learning_rate = 0.05,
        max_depth = 3,
        )
    linear_model = LinearRegression()
    # evaluate models on cross validation
    lin_mae = evaluate_residual_model(linear_model, x, y)
    gb_mae = evaluate_residual_model(gb_model, x, y)
    gb_comp_mae = evaluate_true_distance_model(gb_comp, x, y)
    # save gradient boosting residual model
    gb_model.fit(x, y['residuals'])
    linear_model.fit(x, y['residuals'])
    save_model(linear_model, "linear_model")
    save_model(gb_model, "gb_model")
    # print the baseline residuals
    baseline_mae = (ytst['true_distance'] - xtst['prediction']).abs().mean()
    print(f'linear: {lin_mae}, gradient boosting residuals: {gb_mae}, gradient boosting true: {gb_comp_mae}, baseline: {baseline_mae}')
    
