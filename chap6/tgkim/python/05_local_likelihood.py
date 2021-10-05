import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def main():
    np.random.seed(42)

    N = 200
    k = 30
    x = np.random.rand(N)
    eps = np.random.normal(0, 1/3, N)
    y = np.sin(4*x) + eps
    data = np.column_stack([x, y])

    val_x = np.linspace(0, 1, N//2)
    val_y = np.sin(4*val_x) + np.random.normal(0, 1/3, N//2)

    local_lam_01 = maximum_local_likelihood(data, epanechnikov_quadratic_kernel, 0.1, linear_model)
    local_lam_03 = maximum_local_likelihood(data, epanechnikov_quadratic_kernel, 0.3, linear_model)
    y_lam_01 = local_lam_01(val_x)
    y_lam_03 = local_lam_03(val_x)

    # ==========================================================================
    # Figure
    # ==========================================================================
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Prepare Plot
    plt.figure(figsize=(10,6), dpi=300)
    plt.title(r'Local Likelihood with Epanechnikov & Gaussian Likelihood', fontsize=16)
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$y$', fontsize=14)
    
    # Draw Plot ...
    plt.plot(val_x, val_y, '.', alpha=0.2, label=r'Validate Data')
    plt.plot(val_x, y_lam_01, '-', label=r'$\lambda = 0.1$')
    plt.plot(val_x, y_lam_03, '-', label=r'$\lambda = 0.3$')
    
    # Plot with Legends
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig('figure/05_local_likelihood.png')

def epanechnikov_quadratic_kernel(lam, x0, x):
    t = np.fabs(x0 - x) / lam
    if t <= 1:
        return 0.75 * (1 - t**2)
    else:
        return 0

def log_likelihood(a, b):
    return a * np.log(b) + (1 - a) * np.log(1 - b)

def log_gauss_likelihood(a, b):
    return -(a - b)**2

def linear_model(beta, x_i):
    return beta[0] + beta[1] * x_i

def maximum_local_likelihood(data, kernel, lam, model, likelihood=log_gauss_likelihood):
    def find_beta(x0):
        def evaluation_kernel(x_i):
            return kernel(lam, x0, x_i)

        def local_likelihood(beta):
            s = 0.0
            for x_i, y_i in data:
                s += evaluation_kernel(x_i) * likelihood(y_i, model(beta, x_i))
            return -s
        
        beta = np.random.rand(2)
        res = minimize(local_likelihood, beta, method='L-BFGS-B', tol=1e-6)
        return res.x
    
    def find_y(x):
        beta = find_beta(x)
        return model(beta, x)

    return np.vectorize(find_y)
        
if __name__ == "__main__":
    main()
