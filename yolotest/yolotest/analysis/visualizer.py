import matplotlib.pyplot as plt

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

# plot residuals as a function of true distance 
def 