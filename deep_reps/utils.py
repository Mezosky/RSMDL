from itertools import combinations

import torch


def pca(X, variance_retained=0.99):
    """Perform PCA on the data to retain the specified amount of variance."""
    X_centered = X - X.mean(dim=0)

    # Compute covariance matrix
    covariance_matrix = (X_centered.T @ X_centered) / (X_centered.size(0) - 1)

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = torch.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    # Compute the cumulative explained variance
    explained_variance = eigvals / eigvals.sum()
    cumulative_variance = torch.cumsum(explained_variance, dim=0)

    # Determine the number of components to retain
    num_components = (
        torch.searchsorted(cumulative_variance, variance_retained).item() + 1
    )

    # Project the data onto the principal components
    principal_components = eigvecs[:, :num_components]
    X_pca = X_centered @ principal_components

    return X_pca, principal_components


def vector_to_spd(vector, method="simple_spd"):
    # https://math.stackexchange.com/questions/3717983/generating-symmetric-positive-definite-matrix-from-random-vector-multiplication
    if method == "simple_spd":
        vector = torch.tensor(vector).reshape(-1, 1)
        spd_matrix = torch.mm(vector.T, vector)

    if method == "diagonal":
        vector = torch.tensor(vector).reshape(-1)
        D = torch.diag(vector)
        x, y = D.shape
        U = torch.ones((x, y))
        spd_matrix = D * U * D

    return spd_matrix


def sqrtm_torch(matrix):
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    matrix_sqrt = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
    return matrix_sqrt


def compute_rsm(input_tensor, similarity_function):
    n, _ = input_tensor.shape
    rsm_matrix = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rsm_matrix[i, j] = similarity_function(input_tensor[i], input_tensor[j])
    return rsm_matrix


def normalize_matrix(matrix):
    norm = torch.norm(matrix, p="fro")

    if norm == 0:
        raise ValueError("The norm of the matrix is zero, cannot normalize.")

    normalized_matrix = matrix / norm
    return normalized_matrix


def mean_center_columns(matrix):
    column_means = matrix.mean(dim=0, keepdim=True)
    mean_centered_matrix = matrix - column_means
    return mean_centered_matrix


def matrix_inverse_sqrt(A):
    U, S, V = torch.svd(A)
    S_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(S))
    A_inv_sqrt = U @ S_inv_sqrt @ V.t()
    return A_inv_sqrt


def procrustes(X, Y):
    U, _, V = torch.svd(torch.matmul(X.T, Y))
    Q = torch.matmul(U, V.T)
    return Q


def epsilon_approximate_match(R, R_prime, epsilon):
    _, D = R.shape
    _, D_prime = R_prime.shape
    J_max = set()
    J_prime_max = set()

    for J_size in range(1, min(D, D_prime) + 1):
        for J in combinations(range(D), J_size):
            for J_prime in combinations(range(D_prime), J_size):
                match = True
                for j in J:
                    R_j = R[:, j]
                    min_diff = float("inf")
                    for r in torch.split(R_prime, 1, dim=1):
                        diff = torch.norm(r.squeeze() - R_j)
                        if diff < min_diff:
                            min_diff = diff
                    if min_diff > epsilon * torch.norm(R_j):
                        match = False
                        break

                for j_prime in J_prime:
                    R_prime_j = R_prime[:, j_prime]
                    min_diff = float("inf")
                    for r_prime in torch.split(R, 1, dim=1):
                        diff = torch.norm(r_prime.squeeze() - R_prime_j)
                        if diff < min_diff:
                            min_diff = diff
                    if min_diff > epsilon * torch.norm(R_prime_j):
                        match = False
                        break

                if match:
                    J_max.update(J)
                    J_prime_max.update(J_prime)

    return J_max, J_prime_max


def get_knn_indices_and_ranks(X, k):
    distances = torch.cdist(X, X)
    knn_indices = distances.argsort(dim=1)[:, 1 : k + 1]
    ranks = torch.argsort(torch.argsort(distances, dim=1), dim=1)[:, 1 : k + 1]
    return knn_indices, ranks
