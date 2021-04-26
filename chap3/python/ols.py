import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as ss

# ==============================================================================
# Main
# ==============================================================================
def main():
    np.random.seed(42)

    # Prepare Data to Plot
    x = np.arange(1, 5, 0.1)
    err = np.random.randn(len(x))
    y = 2 * x + 3 + err

    ols = OLSEstimator(x, y)
    ols.estimate()
    ols.test()

    print("beta:", ols.beta_hat)
    print("t_score:", ols.t_score)
    print("p_value:", ols.p_value)

    with plt.xkcd():
        # Prepare Plot
        plt.figure(figsize=(10,6), dpi=300)
        plt.title(r"Simple OLS", fontsize=16)
        plt.xlabel(r'x', fontsize=14)
        plt.ylabel(r'y', fontsize=14)

        # Plot with Legends
        plt.scatter(x, y, label='Data')
        plt.plot(x, np.asarray(ols.y_hat).ravel(), 'r', label='fit')

        # Other options
        plt.legend(fontsize=12)

    plt.savefig("simple_ols.png", dpi=300)


# ==============================================================================
# Estimate
# ==============================================================================
def find_beta_hat(X, y): # X should be np.matrix
    return np.linalg.pinv(X) * y

def find_y_hat(X, beta, y):
    return X * beta

# ==============================================================================
# Test
# ==============================================================================
def find_sigma_hat(y, y_hat, p):
    return np.sum((y - y_hat)**2) / (len(y) - p - 1)

def calc_t_score(beta, X, sigma):
    v = np.sqrt(np.diag(np.linalg.inv(X.T * X)))
    return (beta / v) / np.sqrt(sigma)

def calc_p_value(d, z):
    return (1 - d.cdf(z)) * 2

# ==============================================================================
# OOP Implementation
# ==============================================================================
class OLSEstimator:
    def __init__(self, x, y):
        X = np.matrix(x).T
        X = sm.add_constant(X)
        Y = np.matrix(y).T
        self.X = X
        self.Y = Y
    
    def estimate(self):
        self.beta_hat = find_beta_hat(self.X, self.Y)
        self.y_hat = find_y_hat(self.X, self.beta_hat, self.Y)
    
    def test(self):
        N = self.Y.shape[0]
        p = self.X.shape[1]-1

        self.sigma_hat = find_sigma_hat(np.asarray(self.Y).ravel(), np.asarray(self.y_hat).ravel(), self.X.shape[1]-1)
        self.t_score = calc_t_score(np.asarray(self.beta_hat).ravel(), self.X, self.sigma_hat)
        
        t_dist = ss.t(df=N-p-1)

        p_fun = lambda t: calc_p_value(t_dist, t)
        p_fun = np.vectorize(p_fun)

        self.p_value = p_fun(self.t_score)

if __name__ == "__main__":
    main()