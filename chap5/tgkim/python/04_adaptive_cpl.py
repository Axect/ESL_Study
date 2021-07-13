import numpy as np;
import matplotlib.pyplot as plt
import statsmodels.api as sm

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

    cpl = continuous_piecewise_linear(X, nodes)

    ad_cpl, final_nodes = adaptive_cpl(X, nodes, (0, 10), eps=0.5)
    print(final_nodes)

    # ==========================================================================
    # Plot
    # ==========================================================================
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    # Prepare Plot
    plt.figure(figsize=(10,6), dpi=300)
    plt.title(r'Adaptive Continuous Piecewise Linear', fontsize=16)
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$y$', fontsize=14)
    
    # Draw Plot ...
    plt.scatter(x, y, alpha=0.2, label="data")
    plt.plot(x, ad_cpl(x), color='r', label=r"Adaptive CPL ($\epsilon=5e-1$)")
    
    # Plot with Legends
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig('adaptive_cpl_5e-1.png')

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
   # X = sm.add_constant(x) Error!!!!!!!!
    relus = np.column_stack([relu(x, xi) for xi in nodes])
    return np.column_stack([X, relus])

# Data should be ndarray
def continuous_piecewise_linear(data, nodes):
    X = gen_features(data[:,0], nodes)
    y = data[:,1]
    
    beta = find_beta(X, y)

    print(beta.shape)

    def cpl(x, nodes=nodes):
        X2 = gen_features(x, nodes)
        y = X2 @ beta

        return y

    return cpl

def split_data(data, nodes):
    result = []
    for (a, b) in nodes:
        result.append(data[(data[:,0] >= a) & (data[:,0] < b), :])
    return result

def adaptive_cpl(data, init_nodes, interval, eps=None):
    cpl = continuous_piecewise_linear(data, init_nodes)

    nodes = np.empty((len(init_nodes)+1, 2))
    nodes[0] = (interval[0], init_nodes[0])
    nodes[-1] = (init_nodes[-1], interval[1])
    for i in range(1, len(init_nodes)):
        nodes[i] = (init_nodes[i-1], init_nodes[i])

    splitted = split_data(data, nodes)

    rss_vec = np.empty(len(nodes))

    for i, spd in enumerate(splitted):
        rss_vec[i] = rss(cpl(spd[:,0]), spd[:,1])
    
    mean_rss = np.mean(rss_vec)

    if eps is not None:
        mean_rss = eps
    
    idx = np.where(rss_vec[:] > mean_rss)[0]
    
    offset = 1
    count = 0

    print(mean_rss)

    while len(idx) > 0:
        print(count)
        for i in idx:
            a, b = nodes[offset+i-1]
            a, b, c = np.linspace(a, b, 3)
            nodes[offset+i-1] = (a, b)
            nodes = np.insert(nodes, offset+i, (b, c), axis=0)
            offset += 1
        new_node = np.squeeze(nodes[1:,0]).ravel()
        cpl = continuous_piecewise_linear(data, new_node)
        splitted = split_data(data, nodes)
        # l = len(splitted[0][:,0])
        # if not all(len(s[:,0]) == l for s in splitted):
        #     break
        rss_vec = np.empty(len(nodes))
        for i, spd in enumerate(splitted):
            # print("spd: ", spd.shape)
            # print("node: ", nodes.shape)
            # print("gen: ", gen_features(spd[:,0], new_node).shape)
            rss_vec[i] = rss(cpl(spd[:,0]), spd[:,1])

        idx = np.where(rss_vec[:] > mean_rss)[0]
        offset = 1
        count += 1

    return (cpl, nodes)

if __name__ == "__main__":
    main()