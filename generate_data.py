import numpy as np

def generate_data( num_cluster ):
    n_per = 25
    rng = np.random.default_rng(42)
    X_list = []
    y_list = []
    for k in range(num_cluster):
        t = np.linspace(-1.4, 1.4, n_per)
        if k % 3 == 0:
            phi = np.stack([t, 0.5 * t, 0.2 * t], axis=0)
        elif k % 3 == 1:
            phi = np.stack([t, -0.4 * t, 0.3 * t], axis=0)
        else:
            phi = np.stack([t, 0.1 * t, -0.6 * t], axis=0)
        phi = phi + 0.01 * rng.standard_normal(phi.shape)
        X_list.append(np.tan(phi))
        y_list.append(np.full(n_per, k))
    X = np.hstack(X_list)
    y = np.concatenate(y_list)
    return X, y