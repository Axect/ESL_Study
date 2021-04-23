from netCDF4 import Dataset
import matplotlib.pyplot as plt

# Import netCDF file
ncfile = './data/data.nc'
data = Dataset(ncfile)
var = data.variables

# Use latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Prepare Plot
plt.figure(figsize=(10,6), dpi=300)
plt.title(r"Simple OLS", fontsize=16)
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$y$', fontsize=14)

# Prepare Data to Plot
x = var['x'][:]
y = var['y'][:]  
y_hat = var['y_hat'][:]

# Plot with Legends
plt.scatter(x, y, label=r'$y=2x+3+\varepsilon$')
plt.plot(x, y_hat, 'r', label=r'$\hat{y} = X \hat{\beta}$')

# Other options
plt.legend(fontsize=12)
plt.grid()
plt.savefig("simple_ols.png", dpi=300)
