import torch
import torch.nn.functional as F
import torch.optim as optim

from deep_reps.utils import (
    epsilon_approximate_match,
    matrix_inverse_sqrt,
    normalize_matrix,
    procrustes,
)


def orthogonal_procrustes(R, R_prime):
    R_norm = torch.norm(R, p="fro") ** 2
    R_prime_norm = torch.norm(R_prime, p="fro") ** 2
    product_nuclear_norm = torch.norm(R.T @ R_prime, p="nuc")

    return (R_norm + R_prime_norm - 2 * product_nuclear_norm).item() ** 0.5


def angular_shape_metric(
    R, R_prime, num_iterations=1000, lr=0.01, regularization=1e-20
):
    R = normalize_matrix(R)
    R_prime = normalize_matrix(R_prime)

    _, D = R.shape
    Q = torch.nn.init.orthogonal_(torch.empty(D, D, requires_grad=True))

    optimizer = optim.Adam([Q], lr=lr)
    for _ in range(num_iterations):
        optimizer.zero_grad()
        RQ = R @ Q
        fro_inner_product = torch.norm(RQ * R_prime, p="fro")
        cos_theta = torch.clamp(fro_inner_product, -1.0, 1.0)
        loss = torch.acos(cos_theta)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            U, _, V = torch.svd(
                Q + regularization * torch.eye(Q.shape[0], device=Q.device)
            )
            Q.copy_(torch.matmul(U, V.T))

    return loss.item()


def partial_whitening_shape_metric(
    R, R_prime, alpha=0.5, num_iterations=100, lr=0.01, regularization=1e-20
):
    R = R.clone().detach()
    R_prime = R_prime.clone().detach()
    R = mean_center_columns(R)
    R_prime = mean_center_columns(R_prime)

    N, D = R.shape
    Q = torch.eye(D, requires_grad=True)

    H = torch.eye(N) - 1 / N * torch.ones(N, N)
    func_phi_alpha = lambda X: torch.matmul(H, X) @ (
        alpha * torch.eye(D) + (1 - alpha) * matrix_inverse_sqrt(X.T @ H @ X)
    )
    alpha_R = func_phi_alpha(R)
    alpha_R_prime = func_phi_alpha(R_prime)

    optimizer = optim.Adam([Q], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    for _ in range(num_iterations):
        optimizer.zero_grad()
        alpha_RQ = alpha_R @ Q
        fro_inner_product = torch.norm(alpha_RQ * alpha_R_prime, p="fro")
        fro_alpha_R = torch.norm(alpha_R, p="fro")
        fro_alpha_R_prime = torch.norm(alpha_R_prime, p="fro")

        if (
            torch.isnan(fro_inner_product)
            or torch.isnan(fro_alpha_R)
            or torch.isnan(fro_alpha_R_prime)
        ):
            print("NaN detected, stopping optimization")
            return float("nan")

        cos_theta = torch.clamp(
            fro_inner_product / (fro_alpha_R * fro_alpha_R_prime), -1.0, 1.0
        )
        loss = torch.acos(cos_theta)

        if torch.isnan(loss):
            print("NaN detected in loss, stopping optimization")
            return float("nan")

        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            U, _, V = torch.svd(
                Q + regularization * torch.eye(Q.shape[0], device=Q.device)
            )
            Q.copy_(torch.matmul(U, V.T))

    return loss.item()


def aligned_linear_regression(R, R_prime):
    product_inverse_sqrt = matrix_inverse_sqrt(R_prime.T @ R_prime)
    product_transposed_R = (R_prime @ product_inverse_sqrt).T @ R
    linear_measure = (
        torch.norm(product_transposed_R, p="fro") ** 2 / torch.norm(R, p="fro") ** 2
    )
    return linear_measure.item()


def aligned_cosine_similarity(R, R_prime):
    Q_star = procrustes(R, R_prime)
    RQ = torch.matmul(R, Q_star)

    cos_sims = F.cosine_similarity(RQ.unsqueeze(1), R_prime.unsqueeze(1), dim=2)
    mean_cos_sim = cos_sims.mean()
    return mean_cos_sim.item()


def correlation_match(R, R_prime, epsilon=1e-5):
    _, D = R.shape
    R = R - R.mean(dim=0, keepdim=True)
    R_prime = R_prime - R_prime.mean(dim=0, keepdim=True)

    corr_matrix = torch.matmul(R.T, R_prime)
    _, max_indices = torch.max(corr_matrix, dim=1)
    M = torch.zeros(D, D, device=R.device)
    M[torch.arange(D), max_indices] = 1

    corr_match_sum = 0
    for j in range(D):
        R_j = R[:, j]
        RM_j = torch.matmul(R_prime, M)[:, j]
        norm_R_j = torch.norm(R_j) + epsilon
        norm_RM_j = torch.norm(RM_j) + epsilon
        corr_match_sum += torch.dot(R_j, RM_j) / (norm_R_j * norm_RM_j)

    corr_match = corr_match_sum / D

    return torch.clamp(corr_match, 0, 1.0).item()


def maximum_matching_similarity(R, R_prime, epsilon):
    J_max, J_prime_max = epsilon_approximate_match(R, R_prime, epsilon)
    D = R.shape[1]
    D_prime = R_prime.shape[1]
    return (len(J_max) + len(J_prime_max)) / (D + D_prime)
