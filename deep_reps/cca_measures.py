# implementation in torch
import torch

from deep_reps.utils import pca


def mean_center(X):
    """Mean center the data, due to this method Assume mean-centered representations"""
    return X - X.mean(dim=0)


def compute_covariance_matrices(X, Y):
    n_samples = X.size(0)
    sigma_X = X.T @ X / (n_samples - 1)
    sigma_Y = Y.T @ Y / (n_samples - 1)
    sigma_XY = X.T @ Y / (n_samples - 1)
    return sigma_X, sigma_Y, sigma_XY


def compute_inverse_sqrt_matrix(matrix):
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = torch.clamp(eigvals, min=1e-10)  # Avoid division by zero
    inv_sqrt_matrix = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
    return inv_sqrt_matrix


def compute_cca(X, Y):
    # Center the data
    X_centered = mean_center(X)
    Y_centered = mean_center(Y)

    # Compute covariance matrices
    sigma_X, sigma_Y, sigma_XY = compute_covariance_matrices(X_centered, Y_centered)

    # Compute inverse square roots of covariance matrices
    sqrt_inv_sigma_X = compute_inverse_sqrt_matrix(sigma_X)
    sqrt_inv_sigma_Y = compute_inverse_sqrt_matrix(sigma_Y)

    # Compute the transformation matrix
    T = sqrt_inv_sigma_X @ sigma_XY @ sqrt_inv_sigma_Y

    # Perform SVD on the transformation matrix
    U, S, V = torch.svd(T)

    # Select the top output_dim components
    X_c = X_centered @ sqrt_inv_sigma_X @ U
    Y_c = Y_centered @ sqrt_inv_sigma_Y @ V

    return X_c, Y_c, S


def compute_standard_cca(X, Y):
    _, _, S = compute_cca(X, Y)
    return float(1 / X.size(1) * torch.sum(S))


def compute_yanai_cca(X, Y):
    _, _, S = compute_cca(X, Y)
    return float(1 / X.size(1) * torch.sum(S**2))


def compute_standard_svcca(X, Y, variance_retained=0.99):
    X_pca, _ = pca(X, variance_retained)
    Y_pca, _ = pca(Y, variance_retained)

    _, _, S = compute_cca(X_pca, Y_pca)
    return float(1 / X.size(1) * torch.sum(S))


def compute_yanai_svcca(X, Y, variance_retained=0.99):
    X_pca, _ = pca(X, variance_retained)
    Y_pca, _ = pca(Y, variance_retained)

    _, _, S = compute_cca(X_pca, Y_pca)
    return float(1 / X.size(1) * torch.sum(S**2))


def compute_wvcca(X, Y, output_dim=1):
    X_c, _, S = compute_cca(X, Y, output_dim)

    output_dim = X_c.size(1)
    X = mean_center(X)
    weights = torch.zeros(output_dim)
    for i in range(output_dim):
        Xw = X.T @ X_c[:, i]
        weights[i] = torch.abs(torch.sum(Xw @ X.T))

    # Normalize the weights
    weights = weights / torch.sum(weights)
    # Compute the final PWCCA measure
    return float(torch.sum(weights * S))
