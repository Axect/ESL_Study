import numpy as np;
import matplotlib.pyplot as plt

def main():
    x = np.arange(0, 10, 0.01)
    eps = np.random.randn(len(x))
    y = np.sin(x) + 0.1 * eps

    X = np.column_stack([x, y])
    nodes = np.arange(0, 10, 1)

    pc = piecewise_constant(X, nodes)
    pc = np.vectorize(pc)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    # Prepare Plot
    plt.figure(figsize=(10,6), dpi=300)
    plt.title(r'Piecewise Constant', fontsize=16)
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$y$', fontsize=14)
    
    # Draw Plot ...
    plt.scatter(x, y, alpha=0.2, label="data")
    plt.plot(x, pc(x), color='r', label="piecewise const")
    
    # Plot with Legends
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig('piecewise_const.png')

# Data should be ndarray
def piecewise_constant(data, nodes):
    beta = np.zeros(len(nodes)+1)
    prev_node = nodes[0]
    beta[0] = np.mean(data[data[:,0] < prev_node,1])
    
    for i in range(1, len(nodes)):
        curr_node = nodes[i]
        node_data = data[np.logical_and(data[:,0] >= prev_node, data[:,0] < curr_node), 1]
        beta[i] = np.mean(node_data)
        prev_node = curr_node
    
    beta[-1] = np.mean(data[data[:,0] >= prev_node, 1])

    def pc(x):
        idx = 0
        if x < nodes[0]:
            return beta[0]
        
        for i in range(1, len(nodes)):
            if x < nodes[i]:
                return beta[i]
                
        return beta[-1]
    
    return pc

if __name__ == "__main__":
    main()