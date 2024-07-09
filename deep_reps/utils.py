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
