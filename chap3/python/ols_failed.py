import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Prepare Data to Plot
x = np.arange(1, 10, 0.1)
err = np.random.randn(len(x))
y = x**2 + err

# X = (1 | x)
X = np.matrix(x).T      # To Column matrix
X = sm.add_constant(X)  # Add constant
Y = np.matrix(y).T      # To Column matrix

def find_beta_hat(X, y): # X should be np.matrix
    return np.linalg.pinv(X) * y

def find_y_hat(X, y):
    beta_hat = find_beta_hat(X, y)
    return X * beta_hat

Y_hat = find_y_hat(X, Y)

with plt.xkcd():
    # Prepare Plot
    plt.figure(figsize=(10,6), dpi=300)
    plt.title(r"OLS...?", fontsize=16)
    plt.xlabel(r'x', fontsize=14)
    plt.ylabel(r'y', fontsize=14)
    
    # Plot with Legends
    plt.scatter(x, y, label='Data')
    plt.plot(x, np.squeeze(np.asarray(Y_hat)), 'r', label='fit')
    
    # Other options
    plt.legend(fontsize=12)

plt.savefig("failed_ols.png", dpi=300)

