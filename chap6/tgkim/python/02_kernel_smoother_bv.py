import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

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
    val_data = np.column_stack([val_x, val_y])

    lambdas = np.arange(0.1, 1.0, 0.01)

    sse = np.zeros(len(lambdas))
    variance = np.zeros(len(lambdas))

    for i, lam in enumerate(lambdas):
        g = gen_nadaraya_watson(data, epanechnikov_quadratic_kernel, lam)
        g = np.vectorize(g)
        sse[i] = np.mean((y - np.mean(g(x)))**2)
        variance[i] = np.var(g(x))
    bias = sse - variance
    idx = np.where((np.fabs(variance - bias) == np.min(np.fabs(variance - bias))))[0][0]
    lam = lambdas[idx]
    print("lambda: ", lam)

    g_ref = gen_nadaraya_watson(val_data, epanechnikov_quadratic_kernel, 0.1)
    g_opt = gen_nadaraya_watson(val_data, epanechnikov_quadratic_kernel, lam)
    g_ref2 = gen_nadaraya_watson(val_data, epanechnikov_quadratic_kernel, 0.5)

    y_ref = np.array([g_ref(x) for x in val_data[:,0]])
    y_opt = np.array([g_opt(x) for x in val_data[:,0]])
    y_ref2 = np.array([g_ref2(x) for x in val_data[:,0]])

    r2_ref = r2_score(val_y, y_ref)
    r2_opt = r2_score(val_y, y_opt)
    r2_ref2 = r2_score(val_y, y_ref2)

    print("R2 score (reference): ", r2_ref)
    print("R2 score (optimized): ", r2_opt)
    print("R2 score (reference2): ", r2_ref2)

    # ==========================================================================
    # Figure
    # ==========================================================================
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Prepare Plot
    plt.figure(figsize=(10,6), dpi=300)
    plt.title(r'Bias-Variance Tradeoff for Epanechnikov', fontsize=16)
    plt.xlabel(r'$\lambda$', fontsize=14)
    plt.ylabel(r'Bias or Variance', fontsize=14)
    
    # Draw Plot ...
    plt.plot(lambdas, bias, label=r'Bias')
    plt.plot(lambdas, variance, label=r'Variance')
    
    # Plot with Legends
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig('figure/02_bias_variance.png')

    # Prepare Plot
    plt.figure(figsize=(10,6), dpi=300)
    plt.title(r'Epanechnikov with Various Lambda', fontsize=16)
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$y$', fontsize=14)
    
    # Draw Plot ...
    plt.plot(val_x, val_y, '.', alpha=0.2, label=r'Validate Data')
    plt.plot(val_x, y_ref, label=r'$\lambda=0.1$')
    plt.plot(val_x, y_opt, label=r'$\lambda=%f$' % lam)
    plt.plot(val_x, y_ref2, label=r'$\lambda=0.5$')
    
    # Plot with Legends
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig('figure/02_epanechnikov_diff_lam.png')

def epanechnikov_quadratic_kernel(lam, x0, x):
    t = np.fabs(x0 - x) / lam
    if t <= 1:
        return 0.75 * (1 - t**2)
    else:
        return 0

def gen_nadaraya_watson(data, kernel, lam):
    def nadaraya_watson(x):
        def evaluation_kernel(x_i):
            return kernel(lam, x, x_i)

        denominator = np.sum([evaluation_kernel(x_i) for x_i in data[:,0]])
        numerator = np.sum([evaluation_kernel(x_i) * y_i for (x_i, y_i) in data])

        return numerator / denominator
    return nadaraya_watson

def mse(data, f):
    return np.mean((data[:,1] - np.array([f(x) for x in data[:,0]]))**2)
        
if __name__ == "__main__":
    main()
