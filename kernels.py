import numpy as np


def linear(
        xi: np.ndarray,
        xj: np.ndarray,
        scale: np.ndarray,
) -> np.ndarray:
    assert scale.size >= 3, \
        f"Expected 3 hyperparameters, received {scale.size}"
    if len(xi.shape) > 1 and xi.shape[1] > 1:
        xi_x = xi[:, 0].reshape((-1, 1))
        xj_x = xj[:, 0].reshape((-1, 1))
        linear_x = linear(xi_x, xj_x, scale)

        xi_ic = xi[:, 1].reshape((-1, 1))
        xj_ic = xj[:, 1].reshape((-1, 1))
        linear_ic = linear(xi_ic, xj_ic, scale)

        return linear_x * linear_ic
    return scale[0] + scale[1] * (xi - scale[2]) * (xj.T - scale[2])


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
    if len(xi.shape) > 1 and xi.shape[1] > 1:
        xi = xi[:, 0].reshape((-1, 1))
        xj = xj[:, 0].reshape((-1, 1))
    return scale[0] ** 2 * np.exp(
        2 / scale[1] ** 2 *
        np.sin(np.pi / scale[2] * np.abs(xi - xj.T)) ** 2
    )


def rbf(
        xi: np.ndarray,
        xj: np.ndarray,
        scale: np.ndarray,
) -> np.ndarray:
    assert scale.size >= 2, \
        f"Expected 2 hyperparameters, received {scale.size}"
    if len(xi.shape) > 1 and xi.shape[1] > 1:
        xi = xi[:, 0].reshape((-1, 1))
        xj = xj[:, 0].reshape((-1, 1))
    return scale[0] ** 2 * \
        np.exp(-np.abs(xi - xj.T) ** 2 / (2 * scale[1] ** 2))
