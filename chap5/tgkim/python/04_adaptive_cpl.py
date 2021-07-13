import numpy as np;
import matplotlib.pyplot as plt
import statsmodels.api as sm

def main():
    x = np.arange(0, 10, 0.01)
    eps = np.random.randn(len(x))
    y = np.sin(x) + 0.1 * eps

    X = np.column_stack([x, y])

    nodes = np.arange(0, 10, 1)

    features = gen_features(X[:,0], nodes)
    print(features.shape)

    beta = find_beta(features, X[:,1])
    print(beta.shape)

    new_x = gen_features(np.array([x[0]]), nodes)
    print(new_x.shape)

    print(new_x @ beta)

    cpl = continuous_piecewise_linear(X, nodes)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    # Prepare Plot
    plt.figure(figsize=(10,6), dpi=300)
    plt.title(r'Continuous Piecewise Linear', fontsize=16)
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$y$', fontsize=14)
    
    # Draw Plot ...
    plt.scatter(x, y, alpha=0.2, label="data")
    plt.plot(x, cpl(x), color='r', label="continuous piecewise linear")
    
    # Plot with Legends
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig('continuous_piecewise_linear.png')

def find_beta(features, response):
    X = features
    y = response
    return np.linalg.pinv(X) @ y

def relu(x, xi):
    return np.maximum(x-xi, 0)

def rss(y, y_hat):
    return np.sum((y - y_hat)**2)

def gen_features(x, nodes):
    X = sm.add_constant(x)
    relus = np.column_stack([relu(x, xi) for xi in nodes])
    return np.column_stack([X, relus])

# Data should be ndarray
def continuous_piecewise_linear(data, nodes):
    X = gen_features(data[:,0], nodes)
    y = data[:,1]
    
    beta = find_beta(X, y)

    def cpl(x):
        X2 = gen_features(x, nodes)
        y = X2 @ beta

        return y
    
    return cpl

if __name__ == "__main__":
    main()