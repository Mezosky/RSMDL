import torch

from deep_reps.similarity_functions import euclidean_distance, linear_kernel
from deep_reps.utils import compute_rsm, sqrtm_torch


def rsm_norm_difference(R, R_prime, similarity_function=linear_kernel):
    """RSM Norm Difference"""
    R = R.clone().detach()
    R_prime = R_prime.clone().detach()

    S = compute_rsm(R, similarity_function)
    S_prime = compute_rsm(R_prime, similarity_function)

    return euclidean_distance(S, S_prime).item()


def representation_similarity_analysis(R, R_prime, similarity_function=linear_kernel):
    """Representation Similarity Analysis"""
    R = R.clone().detach()
    R_prime = R_prime.clone().detach()

    S = compute_rsm(R, similarity_function)
    S_prime = compute_rsm(R_prime, similarity_function)

    n, _ = S.shape
    # The lower triangular part of the matrix (v \in \mathbb{R}^{n(n-1)/2})
    v = S[torch.tril_indices(n, n, offset=-1)].flatten()
    v_prime = S_prime[torch.tril_indices(n, n, offset=-1)].flatten()

    return torch.corrcoef(v, v_prime)[0, 1].item()


def centered_kernel_alignment(R, R_prime, similarity_function=linear_kernel):
    """Centered Kernel Alignment"""
    R = R.clone().detach()
    R_prime = R_prime.clone().detach()

    S = compute_rsm(R, similarity_function)
    S_prime = compute_rsm(R_prime, similarity_function)

    return HSIC(S, S_prime) / (HSIC(S, S) * HSIC(S_prime, S_prime)) ** 0.5


def centered_matrix(n):
    return torch.eye(n) - 1 / n * torch.ones((n, n)) @ torch.ones((n, n)).T


def HSIC(S, S_prime):
    n, _ = S.shape
    H = centered_matrix(n)
    return torch.trace(S @ H @ S_prime @ H) * 1 / (n - 1) ** 2


def mean_center(matrix):
    row_means = torch.mean(matrix, dim=1, keepdim=True)
    col_means = torch.mean(matrix, dim=0, keepdim=True)
    grand_mean = torch.mean(matrix)
    centered_matrix = matrix - row_means - col_means + grand_mean

    return centered_matrix


def dCov(S, S_prime):
    """Distance Covariance"""
    n = S.shape[0]
    S_hat = mean_center(S)
    S_prime_hat = mean_center(S_prime)

    return 1 / (n**2) * torch.sum(torch.sum(S_hat @ S_prime_hat, dim=1), dim=0)


def distance_correlation(R, R_prime, similarity_function=euclidean_distance):
    """Distance Correlation"""

    R = R.clone().detach()
    R_prime = R_prime.clone().detach()

    S = compute_rsm(R, similarity_function)
    S_prime = compute_rsm(R_prime, similarity_function)

    return float(
        dCov(S, S_prime) ** 2 / (dCov(S, S) ** 2 * dCov(S_prime, S_prime) ** 2) ** 0.5
    )


def make_positive_semi_definite(matrix, epsilon=1e-10):
    symmetric_matrix = (matrix + matrix.T) / 2
    psd_matrix = symmetric_matrix + epsilon * torch.eye(matrix.shape[0])
    return psd_matrix


def normalized_bures_similarity(R, R_prime, similarity_function=linear_kernel):
    """Normalized Bures Similarity"""
    R = R.clone().detach()
    R_prime = R_prime.clone().detach()

    S = compute_rsm(R, similarity_function)
    S = make_positive_semi_definite(S)
    S_prime = compute_rsm(R_prime, similarity_function)
    S_prime = make_positive_semi_definite(S_prime)
    S_sqrtm = sqrtm_torch(S)
    return float(
        torch.trace(S_sqrtm @ S_prime @ S_sqrtm) ** 0.5
        / (torch.trace(S) * torch.trace(S_prime)) ** 0.5
    )


def eigenspace_overlap_score(R, R_prime, similarity_function=linear_kernel):
    """Eigenspace Overlap Score"""
    R = R.clone().detach()
    R_prime = R_prime.clone().detach()

    S = compute_rsm(R, similarity_function)
    S_prime = compute_rsm(R_prime, similarity_function)

    _, eigmatrix = torch.linalg.eig(S)
    _, eigmatrix_prime = torch.linalg.eig(S_prime)

    denominator = torch.linalg.norm(eigmatrix.T @ eigmatrix_prime, "fro") ** 2
    divisor = max(eigmatrix.shape[1], eigmatrix_prime.shape[1])
    return float(denominator / divisor)


def get_s_minus_lambda(S, lamb=0):
    return torch.linalg.inv(S + lamb * torch.eye(S.shape[0]))


def gulp_similarity(R, R_prime):
    return 1 / R.shape[0] * (R.T @ R_prime)


def unified_linear_probing(R, R_prime, lamb=0.1):
    """Unified Linear Probing"""

    R = R.clone().detach()
    R_prime = R_prime.clone().detach()

    S = gulp_similarity(R, R)
    S_prime = gulp_similarity(R_prime, R_prime)
    S_r_r_prime = gulp_similarity(R, R_prime)

    S_minus_lambda = get_s_minus_lambda(S, lamb)
    S_prime_minus_lambda = get_s_minus_lambda(S_prime, lamb)

    trace_1 = torch.trace(S_minus_lambda @ S @ S_minus_lambda @ S)
    trace_2 = torch.trace(
        S_prime_minus_lambda @ S_prime @ S_prime_minus_lambda @ S_prime
    )
    trace_3 = torch.trace(
        S_minus_lambda @ S_r_r_prime @ S_prime_minus_lambda @ S_r_r_prime.T
    )

    return float((trace_1 + trace_2 - 2 * trace_3) ** 0.5)


def riemannian_similarity(R, R_prime):
    return 1 / R.shape[1] * (R @ R_prime.T)


def riemannian_distance(R, R_prime):
    """Riemannian Distance"""

    R = R.clone().detach()
    R_prime = R_prime.clone().detach()

    S = riemannian_similarity(R, R)
    S_prime = riemannian_similarity(R_prime, R_prime)

    S_inv = torch.linalg.inv(S)
    eigvalue, _ = torch.linalg.eig(S_inv @ S_prime)

    return float(torch.sum(torch.log(eigvalue) ** 2, dim=0) ** 0.5)
