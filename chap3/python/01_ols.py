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

    X = np.matrix(x).T
    X = sm.add_constant(X)
    Y = np.matrix(y).T

    ols = OLSEstimator(X, Y)
    ols.estimate()
    ols.test()

    print("OLS: ")
    ols.summary()

    with plt.xkcd():
        # Prepare Plot
        plt.figure(figsize=(10,6), dpi=300)
        plt.title(r"Simple OLS & Ridge", fontsize=16)
        plt.xlabel(r'x', fontsize=14)
        plt.ylabel(r'y', fontsize=14)

        # Plot with Legends
        plt.scatter(x, y, label='Data')
        plt.plot(x, np.asarray(ols.y_hat).ravel(), 'r', label='OLS')

        # Other options
        plt.legend(fontsize=12)

    plt.savefig("simple_ols.png", dpi=300)

    # Prepare Data to Plot
    x1 = np.arange(1, 5, 0.1)
    x2 = x1**2
    err = np.random.rand(len(x1))
    y = 2 * x2 + 3 * x1 + 5 + 5 * err

    X1 = np.matrix(np.column_stack((x1, x2)))
    X1 = sm.add_constant(X1)
    X2 = np.matrix(x1).T
    X2 = sm.add_constant(X2)
    Y = np.matrix(y).T

    ols_1 = OLSEstimator(X1, Y)
    ols_2 = OLSEstimator(X2, Y)

    ols_1.estimate()
    ols_2.estimate()
    ols_1.test()
    ols_2.test()

    print()
    print("OLS1: ")
    ols_1.summary()

    print()
    print("OLS2: ")
    ols_2.summary()

    f = ols_1.f_test(ols_2)
    print()
    print("F-Test: ", f)

    with plt.xkcd():
        # Prepare Plot
        plt.figure(figsize=(10,6), dpi=300)
        plt.title(r"Two OLS", fontsize=16)
        plt.xlabel(r'x', fontsize=14)
        plt.ylabel(r'y', fontsize=14)

        # Plot with Legends
        plt.scatter(x1, y, label='Data')
        plt.plot(x1, np.asarray(ols_1.y_hat).ravel(), 'r', alpha=0.7, label='OLS1')
        plt.plot(x1, np.asarray(ols_2.y_hat).ravel(), 'g', alpha=0.7, label='OLS2')

        # Other options
        plt.legend(fontsize=12)

    plt.savefig("two_ols.png", dpi=300)

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

def calc_rss(y, y_hat):
    return np.sum((y - y_hat)**2)

def calc_F_score(rss_0, p_0, rss_1, p_1, N):
    return ((rss_0 - rss_1) / (p_1 - p_0)) / (rss_1 / (N - p_1 - 1))

def calc_p_value(d, z):
    if z >= 0:
        return (1 - d.cdf(z)) * 2
    else:
        return 1

# ==============================================================================
# OOP Implementation
# ==============================================================================
class OLSEstimator:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def estimate(self):
        self.beta_hat = find_beta_hat(self.X, self.Y)
        self.y_hat = find_y_hat(self.X, self.beta_hat, self.Y)
    
    def test(self):
        self.N = self.Y.shape[0]
        self.p = self.X.shape[1]-1

        self.sigma_hat = find_sigma_hat(np.asarray(self.Y).ravel(), np.asarray(self.y_hat).ravel(), self.p)
        self.t_score = calc_t_score(np.asarray(self.beta_hat).ravel(), self.X, self.sigma_hat)
        self.rss = calc_rss(np.asarray(self.Y).ravel(), np.asarray(self.y_hat).ravel())
        
        t_dist = ss.t(df=self.N-self.p-1)

        p_fun = lambda t: calc_p_value(t_dist, t)
        p_fun = np.vectorize(p_fun)

        self.p_value = p_fun(self.t_score)
    
    def summary(self):
        print("N: ", self.N)
        print("p: ", self.p)
        print("beta: ", np.asarray(self.beta_hat).ravel())
        print("sigma: ", self.sigma_hat)
        print("t_score: ", self.t_score)
        print("p_value: ", self.p_value)
        print("rss: ", self.rss)

    def f_test(self, other):
        assert self.N == other.N
        p = 0
        f_score = 0.0
        if self.p < other.p:
            f_score = calc_F_score(self.rss, self.p, other.rss, other.p, self.N)
            p = other.p
        elif self.p > other.p:
            f_score = calc_F_score(other.rss, other.p, self.rss, self.p, self.N)
            p = self.p
        
        f_dist = ss.f(abs(self.p - other.p), self.N-p-1)
        p_value = calc_p_value(f_dist, f_score)

        return (f_score, p_value)

if __name__ == "__main__":
    main()
