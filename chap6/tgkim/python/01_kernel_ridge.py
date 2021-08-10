import numpy as np
import matplotlib.pyplot as plt

def find_alpha(K, y, lamb):
    return np.linalg.solve(K + lamb * np.identity(K.shape[0]), y)

def kernel_ridge(K, alpha):
    return K.dot(alpha)

def gaussian_rbf(x, x_m, lamb):
    return np.exp(-lamb * np.sum((x - x_m)**2))

def laplacian_kernel(x, x_m, lamb):
    return np.exp(-lamb * np.sum(np.abs(x - x_m)))

def gen_gram_matrix(X, lamb, kernel):
    N = X.shape[0]
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = kernel(X[i], X[j], lamb)
    return K

def main():
    np.random.seed(42)

    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, x.shape)
    K = gen_gram_matrix(x, 0.1, gaussian_rbf)
    #K = gen_gram_matrix(x, 0.1, laplacian_kernel)
    y_m = kernel_ridge(K, find_alpha(K, y, 0.1))
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Prepare Plot
    plt.figure(figsize=(10,6), dpi=300)
    plt.title(r'Kernel Ridge Regression', fontsize=16)
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$y$', fontsize=14)
    
    # Draw Plot ...
    plt.plot(x, y, 'o', label=r'$y = \sin(x) + \mathcal{N}(0, 0.1)$')
    plt.plot(x, y_m, '-', label=r'$\hat{y} = K\hat{\alpha}$')
    #plt.plot(x, y_m, '-', label='Laplacian Kernel')
    
    # Plot with Legends
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig('figure/01_gaussian_kernel_ridge.png')
    #plt.savefig('figure/01_laplacian_kernel.png')

if __name__ == '__main__':
    main()