from typing import Callable

from copy import deepcopy
from matplotlib import cm, ticker
import matplotlib.pyplot as plt
import numpy as np

from gp import GaussianProcess


class Visualizer:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_test: np.ndarray,
        true_dynamics: Callable,
        mu: np.ndarray,
        cov: np.ndarray,
        gp: GaussianProcess,
        resolution: int = 100,
        n_samples: int = 10,
    ) -> None:
        self.x, self.y = x, y
        self.x_test = x_test
        self.dynamics = true_dynamics
        self.mu, self.cov = mu, cov
        self.gp = deepcopy(gp)
        self.kernel, self.scale = gp.kernel, gp.scale
        self.resolution = resolution
        self.n_samples = n_samples

    def plot_kernel(self) -> None:
        x = np.linspace(
            np.min(self.x), np.max(self.x), self.resolution, True
        )[:, np.newaxis]
        cov_xx = self.kernel(x, x, self.scale)

        fig, ax = plt.subplots()
        im = ax.imshow(cov_xx, origin="lower")
        ax.set_xticks(
            [0, self.resolution],
            [f"{np.min(self.x):.2f}", f"{np.max(self.x):.2f}"]
        )
        ax.set_yticks(
            [0, self.resolution],
            [f"{np.min(self.x):.2f}", f"{np.max(self.x):.2f}"]
        )
        ax.set(frame_on=False)
        ax.set_title(f"{self.kernel.__name__} kernel")
        fig.colorbar(im, ax=ax, label="correlation")
        fig.tight_layout()

    def plot_gp_2d(
            self,
            plot_realizations: bool = False,
            idx: int = None,
    ) -> None:
        if idx is not None:
            x, y = self.x[:, idx], self.y[:, idx]
            x_test, mu = self.x_test[:, idx], self.mu[:, idx]
        else:
            x, y = self.x[:, 0].ravel(), self.y[:, 0].ravel()
            x_test, mu = self.x_test[:, 0].ravel(), self.mu[:, 0].ravel()
        samples = np.random.multivariate_normal(
            mu, self.cov, self.n_samples
        )
        y_test = self.dynamics(x_test)

        fig, ax = plt.subplots()
        ax.plot(x_test, y_test, c="k", ls="dashed", label="ground truth")
        ax.scatter(x, y, c="r", marker="o", label="sampled data")
        ax.plot(x_test, mu, label="predicted mean")
        if plot_realizations:
            for i, sample in enumerate(samples):
                ax.plot(x_test, sample, label=f"gp realization ({i})")

        ci = 1.96 * np.sqrt(np.diag(self.cov))  # 95% ci
        ax.fill_between(
            x_test, mu - ci, mu + ci, alpha=0.2,
            label="95% confidence interval")
        fig.legend()
        fig.tight_layout()

    def plot_3d_ic(
            self,
            ic_range: tuple[float, float],
            idx: int = None,
    ) -> None:
        # fit multivariate gp
        n = self.x.shape[0]
        ic_linspace = np.linspace(ic_range[0], ic_range[1], n, True)
        x_mesh, ic_mesh = np.meshgrid(self.x, ic_linspace)
        x_train = np.vstack([x_mesh.ravel(), ic_mesh.ravel()]).T
        y_mesh = self.dynamics(x_mesh, ic_mesh)
        y_train = y_mesh.ravel()
        self.gp.fit(x_train, y_train)

        # predict y_test
        n_test = self.x_test.shape[0]
        ic_linspace_test = np.linspace(ic_range[0], ic_range[1], n_test, True)
        x_mesh_test, ic_mesh_test = np.meshgrid(self.x_test, ic_linspace_test)
        x_test = np.vstack([x_mesh_test.ravel(), ic_mesh_test.ravel()]).T
        mu, cov = self.gp.predict(x_test)

        # plot ground truth
        y_true_mesh = self.dynamics(x_mesh_test, ic_mesh_test)
        y_true = y_true_mesh.ravel()

        fig_true, ax_true = plt.subplots(subplot_kw={"projection": "3d"})
        ax_true.plot_surface(
            x_mesh_test, ic_mesh_test, y_true_mesh,
        )
        ax_true.set_xlabel("x (time)")
        ax_true.set_ylabel("ic (initial condition)")
        ax_true.set_zlabel("y")
        ax_true.set_title("true dynamics")
        ax_true.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax_true.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        fig_true.tight_layout()

        # plot multivariate gp
        mu_mesh = mu.reshape(x_mesh_test.shape)
        std = np.sqrt(np.diag(cov))
        std = std.reshape(x_mesh_test.shape)
        std_norm = (std - std.min()) / (std.max() - std.min())

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(
            x_mesh_test, ic_mesh_test, mu_mesh, facecolors=cm.viridis(std_norm)
        )
        ax.scatter3D(x_mesh, ic_mesh, y_mesh, c="r", marker="o")
        fig.colorbar(surf, ax=ax, label="covariance")
        ax.set_xlabel("x (time)")
        ax.set_ylabel("ic (initial condition)")
        ax.set_zlabel("y")
        ax.set_title("gaussian process prediction surface")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        fig.tight_layout()

    def show(self) -> None:
        plt.show()
