import numpy as np

from gp import GaussianProcess
from kernels import kernel_cls, linear, quadratic, periodic, rbf

n = 10
x_test = [
    np.zeros((n,)),
    np.zeros((n, 1)),
    np.zeros((n, n)),
    np.zeros((n, n, 1)),
    np.zeros((n, n, n)),
    ]
kernel_test = [linear, quadratic, periodic, rbf]
scale = np.array([1, 1, 1])

for func in kernel_test:
    print(f"\nKernel Test: {func.__name__:<10}")
    kernel = kernel_cls(func)
    print(f"{"Input Shape":<20} {"Output Shape":<20}",
          f" {"Desired Shape":<20} {"Pass":<10}")
    print("-" * 70)
    for x in x_test:
        k = kernel(x, x, scale)
        cond = k.shape == (n, n)
        print(f"{str(x.shape):<20} {str(k.shape):<20}",
              f" {str((n, n)):<20} {str(cond):<10}")

for func in kernel_test:
    print(f"\nGP Test: {func.__name__:<10}")
    kernel = kernel_cls(func)
    gp = GaussianProcess(kernel, scale)
    print(f"{"Input Shape":<20} {"Mu Shape":<20} {"Desired Mu Shape":<20}",
          f" {"Cov Shape":<20} {"Desired Cov Shape":<20} {"Pass":<10}")
    print("-" * 110)
    for x in x_test:
        try:
            gp.fit(x, x)
            mu, cov = gp.predict(x)
            cond = mu.shape == x.shape and cov.shape == (n, n)
        except ValueError:
            cond = False
        print(f"{str(x.shape):<20} {str(mu.shape):<20} {str(x.shape):<20}",
              f" {str(cov.shape):<20} {str((n, n)):<20} {str(cond):<10}")
