from typing import Callable

import numpy as np


class kernel_cls:
    def __init__(self, func: Callable | list[Callable]) -> None:
        self.func = func

    def __call__(
        self,
        xi: np.ndarray,
        xj: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        ni = xi.shape[0]
        nj = xj.shape[0]
        assert xi.shape[1:] == xj.shape[1:], \
            f"Expected dim(xi) == dim(xj), received {xi.shape}, {xj.shape}"
        xi = xi.reshape((ni, -1))
        xj = xj.reshape((nj, -1))
        if isinstance(self.func, list):
            ret = 1
            for f in self.func:
                ret *= f(xi, xj, scale)
            return ret
        return self.func(xi, xj, scale)

    @property
    def __name__(self) -> str:
        if isinstance(self.func, list):
            return "".join([f.__name__ for f in self.func])
        return self.func.__name__


def linear(
        xi: np.ndarray,
        xj: np.ndarray,
        scale: np.ndarray,
) -> np.ndarray:
    assert scale.size >= 3, \
        f"Expected 3 hyperparameters, received {scale.size}"
    return scale[0] + scale[1] * (xi - scale[2]) @ (xj.T - scale[2])


def quadratic(
        xi: np.ndarray,
        xj: np.ndarray,
        scale: np.ndarray,
) -> np.ndarray:
    return linear(xi, xj, scale) ** 2


def periodic(
        xi: np.ndarray,
        xj: np.ndarray,
        scale: np.ndarray,
) -> np.ndarray:
    assert scale.size >= 3, \
        f"Expected 3 hyperparameters, received {scale.size}"
    xi = np.atleast_2d(xi)
    xj = np.atleast_2d(xj)

    if xi.ndim > 2:
        xi = xi.reshape(xi.shape[0], -1)
    if xj.ndim > 2:
        xj = xj.reshape(xj.shape[0], -1)

    dist = np.linalg.norm(xi[:, None] - xj[None, :], axis=2)
    return scale[0] ** 2 * np.exp(
        2 / scale[1] ** 2 *
        np.sin(np.pi / scale[2] * dist) ** 2
    )


def rbf(
        xi: np.ndarray,
        xj: np.ndarray,
        scale: np.ndarray,
) -> np.ndarray:
    assert scale.size >= 2, \
        f"Expected 2 hyperparameters, received {scale.size}"
    sq_dist = np.sum(xi**2, axis=1).reshape(-1, 1) + np.sum(xj**2, axis=1) \
        - 2 * xi @ xj.T
    return scale[0] ** 2 * \
        np.exp(-sq_dist / (2 * scale[1] ** 2))
