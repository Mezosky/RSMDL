import torch
import torch.nn.functional as F

from deep_reps import get_knn_indices_and_ranks


def knn_sim(R, R_prime, k):
    def compute_instance_similarity(
        R_i, R_prime_i, knn_indices_R_i, knn_indices_R_prime_i
    ):
        R_neighbors = R[knn_indices_R_i]
        R_prime_neighbors = R_prime[knn_indices_R_prime_i]

        R_distances = torch.norm(R_i - R_neighbors, dim=1)
        R_prime_distances = torch.norm(R_prime_i - R_prime_neighbors, dim=1)

        return 1 - torch.abs(R_distances - R_prime_distances).mean()

    N, _ = R.shape

    knn_indices_R, _ = get_knn_indices_and_ranks(R, k)
    knn_indices_R_prime, _ = get_knn_indices_and_ranks(R_prime, k)

    similarities = []
    for i in range(N):
        R_i = R[i]
        R_prime_i = R_prime[i]
        knn_indices_R_i = knn_indices_R[i]
        knn_indices_R_prime_i = knn_indices_R_prime[i]

        similarity = compute_instance_similarity(
            R_i, R_prime_i, knn_indices_R_i, knn_indices_R_prime_i
        )
        similarities.append(similarity)

    return torch.tensor(similarities).mean().item()


def knn_jaccard_similarity(R, R_prime, k):
    def compute_jaccard_similarity(knn_indices_R_i, knn_indices_R_prime_i):
        set_R = set(knn_indices_R_i.cpu().numpy())
        set_R_prime = set(knn_indices_R_prime_i.cpu().numpy())
        intersection = len(set_R & set_R_prime)
        union = len(set_R | set_R_prime)
        return intersection / union

    N, _ = R.shape

    knn_indices_R, _ = get_knn_indices_and_ranks(R, k)
    knn_indices_R_prime, _ = get_knn_indices_and_ranks(R_prime, k)

    jaccard_similarities = []
    for i in range(N):
        knn_indices_R_i = knn_indices_R[i]
        knn_indices_R_prime_i = knn_indices_R_prime[i]

        jaccard_similarity = compute_jaccard_similarity(
            knn_indices_R_i, knn_indices_R_prime_i
        )
        jaccard_similarities.append(jaccard_similarity)

    return torch.tensor(jaccard_similarities).mean().item()


def second_order_cosine_similarity(R, R_prime, k):
    def compute_second_order_similarity(R_i_indices, R_prime_i_indices):
        S_i = R[R_i_indices]
        S_prime_i = R_prime[R_prime_i_indices]
        S_i = S_i.view(1, -1)  # Flatten to a single vector
        S_prime_i = S_prime_i.view(1, -1)  # Flatten to a single vector

        cosine_sim = F.cosine_similarity(S_i, S_prime_i)
        return cosine_sim.item()

    N, _ = R.shape

    knn_indices_R, _ = get_knn_indices_and_ranks(R, k)
    knn_indices_R_prime, _ = get_knn_indices_and_ranks(R_prime, k)

    similarities = []
    for i in range(N):
        knn_indices_R_i = knn_indices_R[i]
        knn_indices_R_prime_i = knn_indices_R_prime[i]

        similarity = compute_second_order_similarity(
            knn_indices_R_i, knn_indices_R_prime_i
        )
        similarities.append(similarity)

    return torch.tensor(similarities).mean().item()


def rank_similarity(R, R_prime, k):
    def compute_rank_similarity(i):
        knn_set_R = set(knn_indices_R[i].cpu().numpy())
        knn_set_R_prime = set(knn_indices_R_prime[i].cpu().numpy())
        common_neighbors = knn_set_R & knn_set_R_prime
        K = len(common_neighbors)
        if K == 0:
            return 0.0

        rank_sim = 0
        for j in common_neighbors:
            r_R = (knn_indices_R[i] == j).nonzero(as_tuple=True)[0].item() + 1
            r_R_prime = (knn_indices_R_prime[i] == j).nonzero(as_tuple=True)[
                0
            ].item() + 1
            rank_sim += 1 / ((1 + abs(r_R - r_R_prime)) * (r_R + r_R_prime))

        return 2 * rank_sim / vmax

    N, _ = R.shape

    knn_indices_R, _ = get_knn_indices_and_ranks(R, k)
    knn_indices_R_prime, _ = get_knn_indices_and_ranks(R_prime, k)

    vmax = sum(1 / i for i in range(1, k + 1))

    similarities = []
    for i in range(N):
        similarity = compute_rank_similarity(i)
        similarities.append(similarity)

    return torch.tensor(similarities).mean().item()


def joint_rank_knn_jaccard_similarity(R, R_prime, k):
    rank_sim = rank_similarity(R, R_prime, k)
    jaccard_sim = knn_jaccard_similarity(R, R_prime, k)
    print([rank_sim, jaccard_sim])

    joint_similarity = torch.tensor([rank_sim, jaccard_sim]).mean().item()
    return joint_similarity
