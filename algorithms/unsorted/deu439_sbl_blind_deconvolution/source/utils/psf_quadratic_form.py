import numpy as np

def get_psf_quadratic_form(y, mu, sigma2, m0):
    """
    Represent the norm ||y - theta * x|| as y^T y - 2 theta b + theta^T C theta
    :param y: the observed image (..., N1, N2)
    :param mu: the mean of x (..., N1, N2)
    :param sigma2: the diagonal elements of the covariance of x (..., N1, N2)
    :param m0: the kernel size is M0 = 2*m0 + 1
    :return: b (M0*M0), C (M0*M0, M0*M0)
    """
    # Overall kernel size
    M0 = 2 * m0 + 1

    # Matrix-vector computation we compute the norm as y.T * y - 2*theta.T * b + theta.T * C * theta
    # Shift the image instead of the kernel
    mu = np.roll(np.roll(mu, -m0, axis=-2), -m0, axis=-1)
    sigma2 = np.roll(np.roll(sigma2, -m0, axis=-2), -m0, axis=-1)

    # b vector
    b = np.empty(M0*M0)
    for m in range(0, M0):
        for n in range(0, M0):
            mur = np.roll(np.roll(mu, m, axis=-2), n, axis=-1)
            b[m + n * M0] = np.sum(y * mur)

    # C matrix
    # Mean contribution
    C = np.empty((M0*M0, M0*M0))
    for m1 in range(0, M0):
        for n1 in range(0, M0):
            for m2 in range(0, M0):
                for n2 in range(0, M0):
                    mur1 = np.roll(np.roll(mu, m1, axis=-2), n1, axis=-1)
                    mur2 = np.roll(np.roll(mu, m2, axis=-2), n2, axis=-1)
                    C[m1 + n1*M0, m2 + n2*M0] = np.sum(mur1 * mur2)
    # Covariance contribution
    for m in range(0, M0):
        for n in range(0, M0):
            sigma2r = np.roll(np.roll(sigma2, m, axis=-2), n, axis=-1)
            C[m + n*M0, m + n*M0] += np.sum(sigma2r)

    C = (C.T + C) / 2
    return b, C

def test_get_psf_quadratic_form_deterministic(x):
    from scipy.signal import convolve2d
    import matplotlib.pyplot as plt

    from three_implementations_of_convolution import gkern

    x = x[:40, :50]
    y = x.copy()
    M, N = x.shape
    n0 = 5  # m0 = n0
    theta = gkern(2*n0+1, 1.0)

    # Standard computation of the norm ||y-theta*x||^2
    theta_conv_x = convolve2d(x, theta, boundary='wrap', mode='same')
    res = y - theta_conv_x
    norm = np.sum(res * res)
    print("Standard: ", norm)

    b, C = get_psf_quadratic_form(y, mu=x, sigma2=np.zeros((M, N)), m0=n0)
    theta_vec = theta.ravel('F')
    print("Matrix vector: ", np.sum(y*y) - 2*np.inner(theta_vec, b) + theta_vec.T @ C @ theta_vec)

def test_get_psf_quadratic_form_probabilistic(x):
    from scipy.signal import convolve2d
    import matplotlib.pyplot as plt

    from three_implementations_of_convolution import gkern

    x = x[:40, :50]
    y = x.copy()
    M, N = x.shape
    n0 = 5  # m0 = n0
    theta = gkern(2*n0+1, 1.0)
    sigma2 = np.random.rand(M, N) + 1.0

    # MC computation of the norm ||y-theta*x||^2
    def residual_norm_squared(x_star, theta):
        theta_conv_x = convolve2d(x_star, theta, boundary='wrap', mode='same')
        res = y - theta_conv_x
        return np.sum(res * res)

    n_samples = 200
    norm = 0
    for i in range(n_samples):
        x_star = x + np.sqrt(sigma2)*np.random.randn(M, N)
        norm += residual_norm_squared(x_star, theta)
    print("Monte-Carlo: ", norm / n_samples)

    b, C = get_psf_quadratic_form(y, mu=x, sigma2=sigma2, m0=n0)
    theta_vec = theta.ravel('F')
    print("Closed form: ", np.sum(y*y) - 2*np.inner(theta_vec, b) + theta_vec.T @ C @ theta_vec)

def test_get_psf_quadratic_form1(x):
    from scipy.signal import convolve2d
    import matplotlib.pyplot as plt
    import cvxpy as cp

    from three_implementations_of_convolution import gkern

    M, N = x.shape
    # Build degradation kernel
    n0 = 5  # m0 = n0
    N0 = 2*n0 + 1
    theta_true = gkern(2*n0+1, 1.0)
    y = convolve2d(x, theta_true, boundary='wrap', mode='same')

    yx, yy = np.gradient(y)
    y = np.stack((yx, yy))

    xx, xy = np.gradient(x)
    x = np.stack((xx, xy))

    # Estimate the psf
    b, C = get_psf_quadratic_form(y, mu=x, sigma2=0.001*np.ones((M, N)), m0=n0)
    ones = np.ones(N0*N0)
    theta = cp.Variable(N0*N0)
    problem = cp.Problem(cp.Minimize(cp.quad_form(theta, C) - 2 * theta.T @ b),
                         [theta >= 0, ones.T @ theta == 1])

    problem.solve()
    theta_hat = theta.value
    print(np.sum(theta_hat))

    _, ax = plt.subplots(1, 2)
    ax[0].imshow(theta_true, cmap='gray')
    ax[1].imshow(theta_hat.reshape((N0, N0), order='F'), cmap='gray')
    plt.show()

if __name__ == '__main__':
    import PIL.Image as Image
    from importlib.resources import files

    # Load image
    img_path = files("testing_data") / "im1.png"
    x = np.array(Image.open(img_path)).astype(float) / 255

    # Deterministic test - with zero variance
    test_get_psf_quadratic_form_deterministic(x)

    # Probabilistic test - nonzero variance
    test_get_psf_quadratic_form_probabilistic(x)







