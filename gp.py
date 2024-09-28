from typing import Callable

import numpy as np


class GaussianProcess:
    def __init__(
            self,
            kernel: Callable,
            scale: np.ndarray,
            noise: float = 1e-3
    ) -> None:
        self.kernel = kernel
        self.scale = scale
        self.noise = noise

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x, self.y = x, y
        K = self.kernel(x, x, self.scale) + self.noise * np.eye(len(x))
        self.K_inv = np.linalg.inv(K)

    def predict(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        K_s = self.kernel(self.x, x_test, self.scale)
        K_ss = self.kernel(x_test, x_test, self.scale) + \
            self.noise * np.eye(len(x_test))
        mu_s = K_s.T.dot(self.K_inv).dot(self.y)
        cov_s = K_ss - K_s.T.dot(self.K_inv).dot(K_s)
        return mu_s, cov_s

    def reset(self) -> None:
        del self.x
        del self.y
        del self.K_inv
