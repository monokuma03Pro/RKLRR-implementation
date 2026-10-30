import numpy as np
from scipy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
from sklearn.cluster import SpectralClustering
import pandas as pd

from generate_data import generate_data
from visualization import visualize_3d, plot_heatmap
from functions import phi


def singular_value_shrinkage(X, tau):
    # singular value decomposition: X = U\Sigma V^*
    U, sigma, V_t = svd(X, full_matrices=False) # vector sigma = {\sigma_i} = diagonal elements of matrix Sigma

    # soft-thresholding operation
    D_tau_sigma = np.maximum(sigma - tau, 0.0)
    return (U * D_tau_sigma) @ V_t


def solve_p_subproblem(K, C, lam, rho, N=100):
    n = K.shape[0]
    tau = rho / lam

    eigvals, V = np.linalg.eigh(K) # since K is symmetric and positive semidefinite
    eigvals = np.maximum(eigvals, 0.0)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]

    # the first r_K := rank(K) columns of V
    r_K = np.linalg.matrix_rank(K)
    sigma2 = eigvals[:r_K]
    sigma = np.sqrt(sigma2)
    VK = V[:, :r_K]

    P = np.zeros((n, n))
    eps = 1e-8
    for i in range(n):
        c = C[:, i]
        h_u = VK.T @ c

        if np.linalg.norm(h_u / sigma) <= 1.0 / tau:
            p_i = c - VK @ h_u
        else: 
            # bisection method to find alpha
            def f(alpha):
                denom = tau * alpha + sigma2
                return np.sum((sigma2 * h_u**2) / denom**2) - 1.0 / tau**2

            alpha_lo, alpha_hi = 0.0, 1.0
            while f(alpha_hi) > 0:
                alpha_hi *= 2.0

            for _ in range(N):
                alpha_mid = (alpha_lo + alpha_hi) / 2
                if f(alpha_mid) > 0:
                    alpha_lo = alpha_mid
                else:
                    alpha_hi = alpha_mid
                if alpha_hi - alpha_lo < eps:
                    break
            alpha = 0.5 * (alpha_lo + alpha_hi)

            tmp = sigma2 / (tau * alpha + sigma2)
            p_i = c - VK @ (tmp * h_u)

        P[:, i] = p_i

    return P


def rklrr(K, lam):
    
    n = K.shape[0]

    # initialization
    P = np.eye(n)
    Z = np.zeros((n, n))
    L = np.zeros((n, n))
    rho = 0.5
    rho_max = 1e6
    delta_rho = 0.1
    eps = 1e-5
    N = int(1e6)

    for t in range(N):
        # 1. eq (14) divided by \rho_t
        Z = singular_value_shrinkage( np.eye(n) - P + L / rho, 
                                     tau=1.0 / rho )

        # 2. 
        C = np.eye(n) - Z + L / rho
        P = solve_p_subproblem(K, C, lam, rho)

        # 3.
        L = L + rho * (np.eye(n) - Z - P)

        # 4.
        rho = min(rho * (1 + delta_rho), rho_max)

        # 5.
        err = np.max(np.abs(np.eye(n) - Z - P))
        print(f"iter {t}, err = {err}")
        if err <= eps:
            break

    return Z

def z_to_affinity(Z):
    return np.abs(Z) + np.abs(Z.T)


def spectral_clustering(A, num_cluster):
    sc = SpectralClustering(
        n_clusters=num_cluster,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42,
    )
    return sc.fit_predict(A)


def clustering_error(y_true, y_pred):
    from scipy.optimize import linear_sum_assignment
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    D = max(y_true.max(), y_pred.max()) + 1
    cost = np.zeros((D, D), dtype=int)
    for i in range(len(y_true)):
        cost[y_true[i], y_pred[i]] += 1
    row, col = linear_sum_assignment(-cost)
    acc = cost[row, col].sum() / len(y_true)
    return 1.0 - acc


if __name__ == "__main__":

    num_cluster = 3
    print(f"number of cluster = { num_cluster }")

    # generate data with d = 3
    X, y = generate_data(num_cluster = num_cluster)
    n = X.shape[1]


    # traditional LRR
    print( f"---LRR---" )

    # visualize: X must have d = 3
    visualize_3d(X, y, title=r'$\{\mathbf{x}_i\}_{i=1}^n$'+" in the original input space", save_path='1a.png')

    # obtain K
    K = X.T @ X

    # rklrr
    Z = rklrr(K, lam=0.1)
    plot_heatmap( Z, title=r'Minimizer $\mathbf{Z}^*$ of LRR', save_path='1b.png' )

    A = z_to_affinity(Z)
    y_pred = spectral_clustering(A, num_cluster)
    err = clustering_error(y, y_pred)
    print(f"Clustering error: {err}")


    # RKLRR
    print( f"---RKLRR---" )

    # kernelization
    Phi_X = phi(X)
    # visualize
    visualize_3d(Phi_X, y, title=r'$\{\phi(\mathbf{x}_i)\}_{i=1}^n$'+" in the feature space", save_path='1c.png')

    # obtain K
    K = Phi_X.T @ Phi_X

    # rklrr
    Z = rklrr(K, lam=0.1)
    plot_heatmap( Z, title=r'Minimizer $\mathbf{Z}^*$ of RKLRR', save_path='1d.png' )

    A = z_to_affinity(Z)
    y_pred = spectral_clustering(A, num_cluster)
    err = clustering_error(y, y_pred)
    print(f"Clustering error: {err}")