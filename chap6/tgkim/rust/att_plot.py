import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

# Import parquet file
df = pd.read_parquet('attention.parquet')

# Prepare Data to Plot
att_gaussian = np.array(df['att_gaussian'])
att_epanechnikov = np.array(df['att_epanechnikov'])
att_tricube = np.array(df['att_tricube'])
row = df['row'][0]
col = df['col'][0]

att_gaussian = att_gaussian.reshape((row, col)).T
att_epanechnikov = att_epanechnikov.reshape((row, col)).T
att_tricube = att_tricube.reshape((row, col)).T

atts = [att_gaussian, att_epanechnikov, att_tricube]
labels = ['Gaussian', 'Epanechnikov', 'Tricube']

# Plot
with plt.style.context(["science", "nature"]):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for i, ax in enumerate(axs):
        pcm = ax.imshow(atts[i], cmap='Reds')
        ax.set_title(labels[i])
        ax.set_xlabel('Query')
        ax.set_ylabel('Key')
        ax.set_aspect('equal')
        ax.set_xlim([0, row])
        ax.set_ylim([0, col])
    fig.colorbar(pcm, ax=axs, fraction=0.046, pad=0.04)
    fig.savefig('attention.png', dpi=600, bbox_inches='tight')
