import numpy as np

from dynamics import sin, convex
from gp import GaussianProcess
from kernels import linear, quadratic, periodic, rbf
from visualize import Visualizer

# params
n_train = 50
n_test = 100
dynamics = convex
dynamics_noise_scale = 0
kernel = quadratic
scale = np.array([1, 1, 1])  # [1, 1, 3]
x_range = (-1, 1)
x_test_range = (-1.5, 1.5)
ic_range = (-1, 1)

# sample dynamics
x = np.random.uniform(
    low=x_range[0],
    high=x_range[1],
    size=(n_train, 1)
)
y = dynamics(x) + dynamics_noise_scale * (2 * np.random.random(x.shape) - 1)

# define test points
x_test = np.linspace(
    x_test_range[0],
    x_test_range[1],
    num=n_test,
    endpoint=True,
)[:, np.newaxis]

# fit gp
gp = GaussianProcess(kernel=kernel, scale=scale)
gp.fit(x, y)

# predict y_test
mu_s, cov_s = gp.predict(x_test)

viz = Visualizer(x, y, x_test, dynamics, mu_s, cov_s, gp)
viz.plot_kernel()
viz.plot_gp_2d()
# viz.plot_3d_ic(ic_range=ic_range)
viz.show()
