import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as ss
from sklearn.linear_model import Lars, Ridge, Lasso, lasso_path

# ==============================================================================
# Main
# ==============================================================================
def main():
    np.random.seed(42)

    x = np.arange(0, np.math.pi, 0.01)
    err = np.random.randn(len(x))
    y = np.sin(x) + (x / 3) ** 2 + 0.1 * err
    #y = np.sin(x) + 0.1 * err

    X = np.matrix(np.column_stack([x ** i for i in range(2, 5)] + [np.sin(x)]))
    #X = np.matrix(np.column_stack([x ** i for i in range(2, 6)]))
    #X = np.matrix(np.column_stack([phi(j/10, 1, x) for j in range(31)]))
    Xb = sm.add_constant(X)
    Y = np.matrix(y).T
    X_s = standardize(X)


    # ==============================================================================
    # OLS, Ridge, Lasso
    # ==============================================================================
    ols = OLSEstimator(Xb, Y)
    ols.estimate()
    ols.test()

    ridge = Ridge(alpha=10).fit(X, Y)
    y_ridge = ridge.predict(X)

    lasso = Lasso(alpha=1e-2).fit(X, Y)
    y_lasso = lasso.predict(X)

    # ==============================================================================
    # PCR
    # ==============================================================================
    u, d, vt = np.linalg.svd(X_s, full_matrices=False)
    print(d)
    v = vt.T
    M = 1
    p = v.shape[1]
    z = []
    theta = []
    s = np.zeros((Y.shape[0], 1))
    Y_c = center(Y)

    for i in range(M):
        z_m = X_s * v[:, i] # N x 1
        z.append(z_m)
        theta_m = (z_m.T * Y / (z_m.T * z_m))[0,0]
        theta.append(theta_m)
        s += theta_m * z_m

    y_pcr = s + Y.mean(axis=0)

    # ==============================================================================
    # Plot
    # ==============================================================================
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # Prepare Plot
    plt.figure(figsize=(10,6), dpi=300)
    plt.title(r"OLS, Ridge, Lasso \& PCR", fontsize=16)
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$y$', fontsize=14)

    # Plot with Legends
    plt.scatter(x, y, color="blue", alpha=0.1, label=r'Data')
    plt.plot(x, np.asarray(ols.y_hat).ravel(), color='r', alpha=0.7, label=r'OLS')
    plt.plot(x, y_ridge, color='g', alpha=0.7, label=r'Ridge')
    plt.plot(x, y_lasso, color='purple', alpha=0.7, label=r'Lasso')
    plt.plot(x, y_pcr, color='black', label=r'PCR(1)')

    # Other options
    plt.legend(fontsize=12)
    plt.savefig("pcr_simple_1.png", dpi=300)


# ==============================================================================
# Estimate
# ==============================================================================
def find_beta_hat(X, y, lam=0): # X should be np.matrix
    if lam == 0:
        return np.linalg.pinv(X) * y
    else:
        svd = np.linalg.svd(X, full_matrices=False)
        u, s, vt = svd
        u = np.matrix(u)
        vt = np.matrix(vt)
        s_star = np.matrix(np.diag(s / (s ** 2 + lam)))
        return (vt.T * s_star * u.T * y, svd)

def find_y_hat(X, beta):
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
        return d.cdf(z) * 2

# ==============================================================================
# Scaling
# ==============================================================================
def standardize(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)

def center(y):
    return y - y.mean(axis=0)

# ==============================================================================
# Basis Function
# ==============================================================================
def phi(j, s, x):
    return np.exp(-(x - j) ** 2 / s)

# ==============================================================================
# Coordinate Descent Lasso
# ==============================================================================
def soft_threshold(rho, lam):
    if rho < - lam:
        return rho + lam
    elif rho > lam:
        return rho - lam
    else:
        return 0

def coordinate_descent_lasso(beta, X, y, lam=0.01, num_iters=100):
    m, n = X.shape

    for i in range(num_iters):
        for j in range(n):
            y_hat = X * beta
            rho = X[:,j].T * (y - y_hat + beta[j,0] * X[:,j])

            beta[j] = soft_threshold(rho, lam)
    return beta

# ==============================================================================
# OOP Implementation
# ==============================================================================
class OLSEstimator:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def estimate(self):
        self.beta_hat = find_beta_hat(self.X, self.Y)
        self.y_hat = find_y_hat(self.X, self.beta_hat)
    
    def test(self):
        self.N = self.Y.shape[0]
        self.p = self.X.shape[1]-1
        self.nu = self.N - self.p - 1

        self.sigma_hat = find_sigma_hat(np.asarray(self.Y).ravel(), np.asarray(self.y_hat).ravel(), self.p)
        self.t_score = calc_t_score(np.asarray(self.beta_hat).ravel(), self.X, self.sigma_hat)
        self.rss = calc_rss(np.asarray(self.Y).ravel(), np.asarray(self.y_hat).ravel())
        
        t_dist = ss.t(df=self.nu)

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

class RidgeReg(OLSEstimator):
    def __init__(self, X, Y, lam):
        self.X = standardize(X)
        self.Y = center(Y)
        self.beta_0 = np.mean(Y)
        self.lam = lam

    def estimate(self):
        self.beta_hat, self.svd = find_beta_hat(self.X, self.Y, self.lam)
        self.y_hat = find_y_hat(self.X, self.beta_hat)
        self.true_y_hat = self.y_hat + self.beta_0

    def test(self):
        self.N = self.Y.shape[0]
        self.p = self.X.shape[1]
        (u, s, vt) = self.svd
        s_star = np.matrix(np.diag(s / (s ** 2 + self.lam)))
        
        self.nu = self.N - self.p + np.sum((self.lam / (s ** 2 + self.lam)) ** 2)
        self.rss = calc_rss(np.asarray(self.Y).ravel(), np.asarray(self.y_hat).ravel())
        
        self.sigma_hat = self.rss / self.nu
        v = np.asarray((vt.T * s_star ** 2 * vt).diagonal()).ravel()
        self.t_score = np.asarray(self.beta_hat).ravel() / np.sqrt(self.sigma_hat * v)
        
        t_dist = ss.t(df=self.nu)

        p_fun = lambda t: calc_p_value(t_dist, t)
        p_fun = np.vectorize(p_fun)

        self.p_value = p_fun(self.t_score)

class LassoReg(OLSEstimator):
    def __init__(self, X, Y, lam):
        self.X = X / np.linalg.norm(X, axis=0)
        self.Y = Y
        self.lam = lam
        self.beta = np.matrix(np.ones((X.shape[1], 1)))

    def estimate(self):
        self.beta_hat = coordinate_descent_lasso(self.beta, self.X, self.Y, self.lam)
        self.y_hat = find_y_hat(self.X, self.beta_hat)

if __name__ == "__main__":
    main()
