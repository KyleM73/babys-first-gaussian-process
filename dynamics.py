import numpy as np


def sin(x: np.ndarray, ic: np.ndarray = 0) -> np.ndarray:
    return np.sin(x + ic)


def convex(x: np.ndarray, ic: np.ndarray = 0) -> np.ndarray:
    return (x + ic) ** 2
