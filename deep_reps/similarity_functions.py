import torch


def euclidean_distance(v1, v2):
    return torch.norm(v1 - v2)


def cosine_similarity(v1, v2):
    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))


def linear_kernel(v1, v2):
    return torch.dot(v1, v2) / (v1.size(0) - 1)


def rbf_kernel(v1, v2, sigma=0.1):
    return torch.exp(-(torch.norm(v1 - v2) ** 2) / (2 * sigma**2))


def pearson_correlation(v1, v2):
    v1_mean = torch.mean(v1)
    v2_mean = torch.mean(v2)
    covariance = torch.mean((v1 - v1_mean) * (v2 - v2_mean))
    v1_std = torch.std(v1)
    v2_std = torch.std(v2)
    return covariance / (v1_std * v2_std)


def rbf_kernel(X, Y, sigma=0.1):
    """Radial Basis Function Kernel"""
    return torch.exp(-sigma * torch.norm(X - Y) ** 2)
