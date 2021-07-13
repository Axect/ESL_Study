import numpy as np;
import matplotlib.pyplot as plt

def main():
    # ==========================================================================
    # Generate Sample Data
    # ==========================================================================
    x = np.arange(0, 10, 0.01)
    eps = np.random.randn(len(x))
    y = np.sin(x) + 0.1 * eps

    X = np.column_stack([x, y])

    # ==========================================================================
    # Spline
    # ==========================================================================
    nodes = np.arange(1, 10, 1)

    features = gen_features(X[:,0], nodes)
    print(features.shape)

    beta = find_beta(features, X[:,1])
    print(beta.shape)

    new_x = gen_features(np.array([x[0]]), nodes)
    print(new_x.shape)

    print(new_x @ beta)

    cubic = natural_cubic(X, nodes)

    # ==========================================================================
    # Plot
    # ==========================================================================
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    # Prepare Plot
    plt.figure(figsize=(10,6), dpi=300)
    plt.title(r'Natural Cubic', fontsize=16)
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$y$', fontsize=14)
    
    # Draw Plot ...
    plt.scatter(x, y, alpha=0.2, label="data")
    plt.plot(x, cubic(x), color='r', label=r"Natural Cubic Spline")
    
    # Plot with Legends
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig('natural_cubic.png')

def find_beta(features, response):
    X = features
    y = response
    return np.linalg.pinv(X) @ y

def relu(x, xi):
    return np.maximum(x-xi, 0)

def rss(y, y_hat):
    return np.sum((y - y_hat)**2)

def gen_features(x, nodes):
    bias = np.ones(x.shape[0])
    X = np.column_stack([bias, x])
    d_vec = [(relu(x, xi)**3 - relu(x, nodes[-1])**3) / (nodes[-1] - xi) for xi in nodes[:-1]]
    N = np.column_stack([d - d_vec[-1] for d in d_vec[:-1]])
    return np.column_stack([X, N])

# Data should be ndarray
def natural_cubic(data, nodes):
    X = gen_features(data[:,0], nodes)
    y = data[:,1]
    
    beta = find_beta(X, y)

    print(beta.shape)

    def cubic(x, nodes=nodes):
        X2 = gen_features(x, nodes)
        y = X2 @ beta

        return y

    return cubic

if __name__ == "__main__":
    main()