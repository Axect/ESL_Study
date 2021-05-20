from netCDF4 import Dataset
import matplotlib.pyplot as plt

# Import netCDF file
ncfile = './data/lasso.nc'
data = Dataset(ncfile)
var = data.variables

# Use latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Prepare Plot
plt.figure(figsize=(10,6), dpi=300)
plt.title(r"Linear Regression", fontsize=16)
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$y$', fontsize=14)

# Prepare Data to Plot
x = var['x'][:]
y = var['y'][:]  
y_ols = var['y_ols'][:]
y_ridge = var['y_ridge'][:]
y_lasso = var['y_lasso'][:]

# Plot with Legends
plt.scatter(x, y, label=r'Data')
plt.plot(x, y_ols, 'r', label=r'OLS')
plt.plot(x, y_ridge, 'g', label=r'Ridge')
plt.plot(x, y_lasso, color='purple', label=r'Lasso', alpha=0.5)

# Other options
plt.legend(fontsize=12)
plt.grid()
plt.savefig("lasso.png", dpi=300)
