import numpy as np
import matplotlib.pyplot as plt

def main():
    np.random.seed(42)

    N = 200
    k = 30
    x = np.random.rand(N)
    eps = np.random.normal(0, 1/3, N)
    y = np.sin(4*x) + eps
    data = np.column_stack([x, y])

    # idx = knn(x, 0, 30)
    # print(idx)

    f = gen_knn_average(data, k)
    f = np.vectorize(f)

    g = gen_nadaraya_watson(data, epanechnikov_quadratic_kernel, 0.2)
    h = local_weighted_linear(data, epanechnikov_quadratic_kernel, 0.2)

    domain = np.linspace(0, 1, 100)
    knn_image = f(domain)
    eq_image = np.array([g(x0) for x0 in domain])
    lw_image = np.array([h(x0) for x0 in domain])

    knn_mse = mse(data, f)
    eq_mse = mse(data, g)
    lw_mse = mse(data, h)

    # ==========================================================================
    # Figure
    # ==========================================================================
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Prepare Plot
    plt.figure(figsize=(10,6), dpi=300)
    plt.title(r'Various kernel smoother', fontsize=16)
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$y$', fontsize=14)
    
    # Draw Plot ...
    plt.scatter(x, y, alpha=0.2, label="Data")
    plt.plot(domain, np.sin(4*domain), alpha=0.5, label="original")
    plt.plot(domain, knn_image, alpha=0.8, label="kNN")
    plt.plot(domain, eq_image, alpha=0.8, label="Epanechnikov quadratic")
    plt.plot(domain, lw_image, label="Local weighted linear")
    
    # Plot with Legends
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig('figure/03_local_weighted_linear.png')

    # ==========================================================================
    # Report
    # ==========================================================================
    print("kNN MSE:", knn_mse)
    print("Epanechnikov quadratic MSE:", eq_mse)
    print("Local weighted linear MSE:", lw_mse)

def knn(arr, x, k):
    dist = np.fabs(arr - x)
    idx = np.argsort(dist)
    return idx[:k]

def gen_knn_average(data, k):
    def knn_average(x):
        idx = knn(data[:,0], x ,k)
        ave = np.mean(np.take(data[:,1], idx))
        return ave
    return knn_average

def epanechnikov_quadratic_kernel(lam, x0, x):
    t = np.fabs(x0 - x) / lam
    if t <= 1:
        return 0.75 * (1 - t**2)
    else:
        return 0

def tri_cube(lam, x0, x):
    t = np.fabs(x0 - x) / lam
    if t <= 1:
        return (1-t**3)**3
    else:
        return 0

def gaussian_kernel(lam, x0, x):
    t = np.fabs(x0 - x) / lam
    return np.exp(-t**2 / 2)

def gen_nadaraya_watson(data, kernel, lam):
    def nadaraya_watson(x):
        def evaluation_kernel(x_i):
            return kernel(lam, x, x_i)

        denominator = np.sum([evaluation_kernel(x_i) for x_i in data[:,0]])
        numerator = np.sum([evaluation_kernel(x_i) * y_i for (x_i, y_i) in data])

        return numerator / denominator
    return nadaraya_watson

def local_weighted_linear(data, kernel, lam):
    x = data[:,0]
    y = data[:,1].reshape(-1, 1)
    N = len(x)
    
    def f_hat(x0):
        bT = np.array([[1, x0]])
        B = np.column_stack([np.ones(N), x])
        W = np.diag(np.array([kernel(lam, x0, x_i) for x_i in x]))
        return (bT @ np.linalg.inv(B.T @ W @ B) @ B.T @ W @ y)[0,0]

    return f_hat
        
def mse(data, f):
    return np.mean((data[:,1] - np.array([f(x) for x in data[:,0]]))**2)


if __name__ == "__main__":
    main()
