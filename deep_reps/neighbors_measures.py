import torch


def knn_sim(R, R_prime, k):
    N, D = R.shape
    
    def get_knn_indices(X, k):
        distances = torch.cdist(X, X)
        knn_indices = distances.argsort(dim=1)[:, 1:k+1]
        return knn_indices

    knn_indices_R = get_knn_indices(R, k)
    knn_indices_R_prime = get_knn_indices(R_prime, k)
    
    def compute_instance_similarity(R_i, R_prime_i, knn_indices_R_i, knn_indices_R_prime_i):
        R_neighbors = R[knn_indices_R_i]
        R_prime_neighbors = R_prime[knn_indices_R_prime_i]
        
        R_distances = torch.norm(R_i - R_neighbors, dim=1)
        R_prime_distances = torch.norm(R_prime_i - R_prime_neighbors, dim=1)
        
        return 1 - torch.abs(R_distances - R_prime_distances).mean()
    
    similarities = []
    for i in range(N):
        R_i = R[i]
        R_prime_i = R_prime[i]
        knn_indices_R_i = knn_indices_R[i]
        knn_indices_R_prime_i = knn_indices_R_prime[i]
        
        similarity = compute_instance_similarity(R_i, R_prime_i, knn_indices_R_i, knn_indices_R_prime_i)
        similarities.append(similarity)
    
    return torch.tensor(similarities).mean().item()