import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def main():
    np.random.seed(42)

    lam = 0.01

    N = 200
    k = 30
    x = np.random.rand(N)
    eps = np.random.normal(0, 1/3, N)
    y = x*np.sin(4*x) + 0.1*eps
    data = np.column_stack([x, y])

    f_natural = gen_natural_local_estimate(x, lam)
    phi_lam = gen_phi_lam(lam)
    f_parzen = gen_parzen_estimate(x, phi_lam, lam)
    
    domain = np.linspace(0, 1, 1000)
    y_natural = f_natural(domain)
    y_parzen = f_parzen(domain)

    # ==========================================================================
    # Figure
    # ==========================================================================
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Prepare Plot
    plt.figure(figsize=(10,6), dpi=300)
    plt.title(r'KDE $(\lambda = ' + f"{lam}" + r")$", fontsize=16)
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$f(x)$', fontsize=14)
    
    # Draw Plot ...
    plt.plot(x, 0*x, '.', label=r'$x$')
    plt.plot(domain, y_natural / np.sum(y_natural), label='Natural')
    plt.plot(domain, y_parzen / np.sum(y_parzen), label='Parzen')
    
    # Plot with Legends
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig(f'figure/04_kde_{lam}.png')


def knn_lam(arr, x, lam):
    dist = np.fabs(arr - x)
    idx = np.argsort(dist)
    dist_sorted = dist[idx]
    return idx[np.where(dist_sorted < lam)]

def gen_natural_local_estimate(x, lam):
    def f(x0):
        x0_nearest = knn_lam(x, x0, lam)
        return len(x0_nearest) / (len(x) * lam)
    return np.vectorize(f)

def gen_parzen_estimate(x, kernel, lam):
    def f(x0):
        return np.sum(np.array([kernel(x0, xi) for xi in x])) / (len(x) * lam)
    return np.vectorize(f)

def gen_phi_lam(lam):
    def k(x0, xi):
        return np.exp(-np.linalg.norm(xi - x0) ** 2 / lam**2) / np.sqrt(2 * np.pi)
    return k

if __name__ == '__main__':
    main()