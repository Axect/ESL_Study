import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

# Import parquet file
df = pd.read_parquet('nadaraya_watson.parquet')

# Prepare Data to Plot
x_data = df['x_data']
y_data = df['y_data']
x = df['x']
y = df['y']
gaussian = df['gaussian']
epanechnikov = df['epanechnikov']
tricube = df['tricube']

data = np.column_stack([x_data, y_data])
# sort by x
data = data[np.argsort(data[:, 0])]
x_data = data[:, 0]
y_data = data[:, 1]

y_hat = [gaussian, epanechnikov, tricube]
labels = ['Gaussian', 'Epanechnikov', 'Tricube']

# Plot params
pparam = dict(
    xlabel = r'$x$',
    ylabel = r'$y$',
)

# Plot
with plt.style.context(["science", "nature"]):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    for i, ax in enumerate(axs):
        ax.autoscale(tight=True)
        ax.set(**pparam)
        ax.plot(x_data, y_data, '.', alpha=0.3, label='Data')
        ax.plot(x, y, label=r'True')
        ax.plot(x, y_hat[i], label=labels[i])
        ax.legend()
    fig.savefig('nadaraya_watson.png', dpi=600, bbox_inches='tight')
