import torch
from deep_reps.utils import pca

class BaseCCA:
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        self.X_raw = X
        self.Y_raw = Y
        self.X = self._mean_center(X)
        self.Y = self._mean_center(Y)
        self.sigma_X, self.sigma_Y, self.sigma_XY = self._compute_covariances()

    def _mean_center(self, X):
        return X - X.mean(dim=0)

    def _compute_covariances(self):
        n_samples = self.X.size(0)
        sigma_X = self.X.T @ self.X / (n_samples - 1)
        sigma_Y = self.Y.T @ self.Y / (n_samples - 1)
        sigma_XY = self.X.T @ self.Y / (n_samples - 1)
        return sigma_X, sigma_Y, sigma_XY

    def _compute_inv_sqrt(self, matrix):
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        eigvals = torch.clamp(eigvals, min=1e-10)
        return eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T

    def compute_cca(self, output_dim=None):
        inv_sqrt_X = self._compute_inv_sqrt(self.sigma_X)
        inv_sqrt_Y = self._compute_inv_sqrt(self.sigma_Y)
        T = inv_sqrt_X @ self.sigma_XY @ inv_sqrt_Y
        U, S, Vh = torch.linalg.svd(T)
        V = Vh.T

        if output_dim is not None:
            U = U[:, :output_dim]
            S = S[:output_dim]
            V = V[:, :output_dim]

        self.S = S
        self.X_proj = self.X @ inv_sqrt_X @ U
        self.Y_proj = self.Y @ inv_sqrt_Y @ V
        return self.X_proj, self.Y_proj, S


class StandardCCA(BaseCCA):
    def compute_similarity(self):
        _, _, S = self.compute_cca()
        return float(S.mean())


class YanaiCCA(BaseCCA):
    def compute_similarity(self):
        _, _, S = self.compute_cca()
        return float(torch.mean(S ** 2))


class SVCCA(BaseCCA):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, variance_retained=0.99):
        X_pca, _ = pca(X, variance_retained)
        Y_pca, _ = pca(Y, variance_retained)
        super().__init__(X_pca, Y_pca)

    def compute_similarity(self):
        _, _, S = self.compute_cca()
        return float(S.mean())


class YanaiSVCCA(BaseCCA):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, variance_retained=0.99):
        X_pca, _ = pca(X, variance_retained)
        Y_pca, _ = pca(Y, variance_retained)
        super().__init__(X_pca, Y_pca)

    def compute_similarity(self):
        _, _, S = self.compute_cca()
        return float(torch.mean(S ** 2))


class PWCCA(BaseCCA):
    def compute_similarity(self, output_dim=1):
        X_proj, _, S = self.compute_cca(output_dim)
        output_dim = X_proj.size(1)

        X_centered = self._mean_center(self.X_raw)
        weights = torch.zeros(output_dim)
        for i in range(output_dim):
            Xw = X_centered.T @ X_proj[:, i]
            weights[i] = torch.abs(torch.sum(Xw @ X_centered.T))

        weights /= weights.sum()
        return float(torch.sum(weights * S))