import numpy as np;
import matplotlib.pyplot as plt
import statsmodels.api as sm

def main():
    x = np.arange(0, 10, 0.01)
    eps = np.random.randn(len(x))
    y = np.sin(x) + 0.1 * eps

    X = np.column_stack([x, y])
    X = sm.add_constant(X)
    nodes = np.arange(0, 10, 1)

    pl = piecewise_linear(X, nodes)
    pl = np.vectorize(pl)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    # Prepare Plot
    plt.figure(figsize=(10,6), dpi=300)
    plt.title(r'Piecewise Linear', fontsize=16)
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$y$', fontsize=14)
    
    # Draw Plot ...
    plt.scatter(x, y, alpha=0.2, label="data")
    plt.plot(x, pl(x), color='r', label="piecewise linear")
    
    # Plot with Legends
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig('piecewise_linear.png')

def find_beta(data):
    return np.linalg.pinv(data[:,0:2]) @ data[:,2]

# Data should be ndarray
def piecewise_linear(data, nodes):
    beta = np.zeros((len(nodes)+1, 2))
    prev_node = nodes[0]
    beta[0,:] = find_beta(data[data[:,1] < prev_node,:])
    
    for i in range(1, len(nodes)):
        curr_node = nodes[i]
        node_data = data[(data[:,1] >= prev_node) & (data[:,1] < curr_node), :]
        beta[i,:] = find_beta(node_data)
        prev_node = curr_node
    
    beta[-1,:] = find_beta(data[data[:,1] >= prev_node, :])

    def pl(x):
        idx = 0
        if x < nodes[0]:
            return beta[0,:][0] + beta[0,:][1] * x
        
        for i in range(1, len(nodes)):
            if x < nodes[i]:
                return beta[i,:][0] + beta[i,:][1] * x
                
        return beta[-1,:][0] + beta[-1,:][1] * x
    
    return pl

if __name__ == "__main__":
    main()