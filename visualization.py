import numpy as np
import matplotlib.pyplot as plt

colors = ['#1f77b4', '#2ca02c', '#d62728',
          '#9467bd', '#ff7f0e', '#8c564b',
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
markers = ['o', '^', 's', 'D', 'v', 'P', '*', 'X', '<', '>']

def visualize_3d( X, y, num_cluster=None, title=None, save_path=None ):
    X = np.asarray(X)
    y = np.asarray(y)

    if num_cluster is None:
        num_cluster = int(y.max()) + 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for k in range(num_cluster):
        idx = (y == k)
        if np.sum(idx) == 0:
            continue
        ax.scatter(
            X[0, idx], X[1, idx], X[2, idx], # choose points with idx = True, i.e. in the k-th cluster
            c=colors[k], marker=markers[k],
            s=30, alpha=0.8,
        )
    if title is not None:
        ax.set_title(title)
    ax.view_init(elev=10, azim=45)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    return fig, ax


def plot_heatmap(
    Z,
    title="",
    save_path=None,
):
    Z = np.asarray(Z)
    fig, ax = plt.subplots()
    im = ax.imshow(
        Z,
        cmap="jet",
        aspect="equal",
        interpolation="nearest",
        vmin=0,
        vmax=0.07,
        origin="upper",
        extent=[0, 75, 75, 0]
    )

    cbar = fig.colorbar( im, ax=ax, fraction=0.046, pad=0.04 )

    ax.set_title( title )

    if save_path is not None:
        fig.savefig( save_path )

    return fig, ax