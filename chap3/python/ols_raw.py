import numpy as np
import statsmodels.api as sm
import scipy.stats as ss

def find_beta_hat(X, y): # X should be np.matrix
    return np.linalg.pinv(X) * y

def find_y_hat(X, beta, y):
    return X * beta

def find_sigma_hat(y, y_hat, p):
    return np.sum((y - y_hat)**2) / (len(y) - p - 1)

def calc_t_score(beta, X, sigma):
    v = np.sqrt(np.diag(np.linalg.inv(X.T * X)))
    return (beta / v) / np.sqrt(sigma)

def calc_rss(y, y_hat):
    return np.sum((y - y_hat)**2)

def calc_F_score(rss_0, p_0, rss_1, p_1, N):
    return ((rss_0 - rss_1) / (p_1 - p_0)) / (rss_1 / (N - p_1 - 1))

def calc_p_value(d, z):
    return (1 - d.cdf(z)) * 2

# Example
np.random.seed(42)
x = np.arange(1, 5, 0.1)
err = np.random.randn(len(x))
y = 2 * x + 3 + err

X = np.matrix(x).T
X = sm.add_constant(X)
Y = np.matrix(y).T

beta = find_beta_hat(X, Y)
y_hat = find_y_hat(X, beta, Y)

print("beta: ", np.asarray(beta).ravel())
print("y_hat: ", np.asarray(y_hat).ravel())