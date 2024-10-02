import numpy as np


def sin(x: np.ndarray, ic: np.ndarray = 0) -> np.ndarray:
    return np.sin(np.pi * (x + ic))


def exp_decay(x: np.ndarray, ic: np.ndarray = 0) -> np.ndarray:
    return np.exp(-(x + ic))


def exp_decay_sin(x: np.ndarray, ic: np.ndarray = 0) -> np.ndarray:
    return exp_decay(x, ic) * sin(x, ic)


def convex(x: np.ndarray, ic: np.ndarray = 0) -> np.ndarray:
    return (x + ic) ** 2
