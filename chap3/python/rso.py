import numpy as np
import statsmodels.api as sm

def calc_gamma(z, x):
    return np.dot(z, x) / np.dot(z, z)

def calc_z(X):  # X is np.array (not matrix)
    p = X.shape[1] - 1
    z = [X[:, 0]]

    for j in range(1, p+1):
        x = X[:, j]
        s = np.zeros(len(x))
        for l in range(j):
            g = calc_gamma(z[l], x)
            s += g * z[l]
        z.append(x - s)

    return z

def calc_beta(y, Z): # calculate beta_1, beta_2, ...
    beta = []
    for z in Z:
        b = calc_gamma(z, y)
        beta.append(b)
    return beta[1:]

def calc_intercept(X, y, beta): # calculate beta_0
    y_bar = np.mean(y)
    X_bar = np.mean(X[:,1:], axis=0)
    return y_bar - np.dot(X_bar, beta)

x = np.arange(1, 5, 0.1)
err = np.random.randn(len(x))
y = 2 * x + 5 + err

X = np.array(x).T
X = sm.add_constant(X)
Y = np.array(y).T

z = calc_z(X)
beta = calc_beta(Y, z)
print("slope: ", beta)
beta_0 = calc_intercept(X, y, beta)
print("intercept: ", beta_0)
