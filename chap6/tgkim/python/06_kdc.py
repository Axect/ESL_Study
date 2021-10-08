import numpy as np
import matplotlib.pyplot as plt

def main():
    np.random.seed(42)

    lam = 5

    N = 100

    math1 = np.random.normal(70, 10, N)
    math2 = np.random.normal(40, 15, N)
    # eng1 = np.random.normal(50, 10, N)
    # eng2 = np.random.normal(80, 5, N)

    # data1 = np.column_stack([math1, eng1])
    # data2 = np.column_stack([math2, eng2])

    # 1. KDC for math (Natural Local Estimate)
    f_natural_1 = gen_natural_local_estimate(math1, lam)
    f_natural_2 = gen_natural_local_estimate(math2, lam)
    f_natural_vec = np.array([f_natural_1, f_natural_2])
    pi_vec = np.array([0.5, 0.5])
    p_vec = gen_kdc(f_natural_vec, pi_vec)

    domain = np.arange(0, 100, 0.01)
    print(p_vec(domain[5000]))
    p_nat_math = np.array([p_vec(x) for x in domain])
    print(p_nat_math)

    # 2. KDC for math (Parzen Estimate)
    kernel = gen_phi_lam(lam)
    f_parzen_1 = gen_parzen_estimate(math1, kernel, lam)
    f_parzen_2 = gen_parzen_estimate(math2, kernel, lam)
    f_parzen_vec = np.array([f_parzen_1, f_parzen_2])
    pi_vec = np.array([0.5, 0.5])
    p_vec = gen_kdc(f_parzen_vec, pi_vec)

    domain = np.arange(0, 100, 0.01)
    print(p_vec(domain[5000]))
    p_parzen_math = np.array([p_vec(x) for x in domain])
    print(p_parzen_math)

    # ==========================================================================
    # Figure
    # ==========================================================================
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # 0. Histogram for math
    plt.figure(figsize=(8, 6))
    plt.title(r'Histogram for math')
    plt.hist(math1, bins=20, density=True, alpha=0.5, label='Group1')
    plt.hist(math2, bins=20, density=True, alpha=0.5, label='Group2')
    plt.xlabel(r'Math Score')
    plt.ylabel(r'Density')
    plt.legend()
    plt.savefig('figure/06_kdc_hist.png')

    # 1. Figure for natural local estimate
    # Prepare Plot
    plt.figure(figsize=(10,6), dpi=300)
    plt.title(r'KDC with Natural $(\lambda = ' + f"{lam}" + r")$", fontsize=16)
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'${\rm Pr}(G=j|X=x)$', fontsize=14)
    
    # Draw Plot ...
    plt.plot(domain, p_nat_math[:,0], label='Group1')
    plt.plot(domain, p_nat_math[:,1], label='Group2')
    
    # Plot with Legends
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig(f'figure/06_kdc_natural.png')

    # 2. Figure for parzen estimate
    # Prepare Plot
    plt.figure(figsize=(10,6), dpi=300)
    plt.title(r'KDC with Parzen $(\lambda = ' + f"{lam}" + r")$", fontsize=16)
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'${\rm Pr}(G=j|X=x)$', fontsize=14)

    # Draw Plot ...
    plt.plot(domain, p_parzen_math[:,0], label='Group1')
    plt.plot(domain, p_parzen_math[:,1], label='Group2')

    # Plot with Legends
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig(f'figure/06_kdc_parzen.png')

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
        return np.exp(-np.linalg.norm(xi - x0) ** 2 / (2*lam**2)) / np.sqrt(2 * np.pi)
    return k

def gen_kdc(f_vec, pi_vec):
    def p_vec(x0):
        arr = np.array([pi_vec[i] * f_vec[i](x0) for i in range(len(f_vec))])
        s = np.sum(arr)
        return arr / s if s > 0 else np.zeros_like(arr)
    return p_vec

if __name__ == '__main__':
    main()