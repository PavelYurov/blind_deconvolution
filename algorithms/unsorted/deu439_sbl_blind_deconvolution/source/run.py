import numpy as np
import cvxpy as cp
from importlib.resources import files

from .utils.custom_cg import conjugate_grad
from .utils.psf_quadratic_form import get_psf_quadratic_form


def update_qx(y, theta, gamma, lam):
    """
    Update the parameters of the variational pdf q(x)=N(x; mu, sigma2)
    :param y: the observed image (K, N1, N2)
    :param theta: the point-spread function (M0, M0)
    :param gamma: the varitation parameters (K, N1, N2)
    :param lam: measurement noise precision (scalar)
    :return: mu (K, N1, N2), sigma2 (K, N1, N2)
    """
    m0 = theta.shape[0] // 2
    channels, height, width = y.shape

    # Pad and shift the point-spread function
    vertical_padding = height - theta.shape[0]
    horizontal_padding = width - theta.shape[1]
    theta_pad = np.pad(theta, [[0, vertical_padding], [0, horizontal_padding]])
    theta_pad = np.roll(theta_pad, -m0, axis=0)
    theta_pad = np.roll(theta_pad, -m0, axis=1)

    # Calculate b = A(theta)^T y
    Theta = np.fft.fft2(theta_pad)
    Y = np.fft.fft2(y)
    b = np.real(np.fft.ifft2(Y * np.conj(Theta)))

    # Solve (lam A(theta)^T A(theta) + Gamma)mu = lam A(theta)^T y using custom conjugate gradient method
    mu = conjugate_grad(theta_pad, gamma, lam, lam*b, x=None)

    # Closed form for the diagonal covariance matrix
    sigma2 = 1 / (lam * np.sum(theta * theta) + gamma)

    return mu, sigma2

def update_theta(y, mu, sigma2, m0):
	"""
	Update the point-spread function estimate
	:param y: observed image (N1, N2)
	:param mu: the mean of q(x) (N1, N2)
	:param sigma2: the diagonal of the covariance of q(x) (N1, N2)
	:param m0: the point-spread function size is 2*m0 + 1
	:return: theta (M0, M0)
	"""
	M0 = 2*m0 + 1
	b, C = get_psf_quadratic_form(y, mu, sigma2, m0=m0)
	C = 0.5 * (C + C.T)
	C = C.astype(np.float64)
	P = cp.psd_wrap(C)  # если уверен в PSD по теории

	theta = cp.Variable(C.shape[0])
	obj = cp.Minimize(cp.quad_form(theta, P) - 2 * b.T @ theta)
	cons = [theta >= 0, cp.sum(theta) == 1]

	prob = cp.Problem(obj, cons)
	prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, max_iter=200000)
	return theta.value.reshape(M0, M0, order='F')

def update_gamma(mu, sigma2):
    """
    Update the variational parameter gamma.
    :param mu: the mean of q(x) (N1, N2)
    :param sigma2: the diagonal of the covariance of q(x) (N1, N2)
    :return: gamma (N1, N2)
    """
    return 1 / (mu*mu + sigma2)


def run(y, m0, lam=1.0, n_iter=20):
    # Initialize theta, and gamma
    M0 = 2*m0 + 1
    theta = np.random.rand(M0, M0)
    theta = theta / np.sum(theta)
    gamma = np.ones_like(y)

    _, ax = plt.subplots(2, 2)
    for i in range(n_iter):
        mu, sigma2 = update_qx(y, theta, gamma, lam)
        theta = update_theta(y, mu, sigma2, m0)
        gamma = update_gamma(mu, sigma2)

        ax[0,0].imshow(theta, cmap='gray')
        ax[0,1].imshow(mu[0], cmap='gray')
        ax[1,0].imshow(mu[1], cmap='gray')
        ax[1,1].imshow(sigma2[0], cmap='gray')
        plt.pause(1.0)
        print(f"Iteration {i}/{n_iter}")

    return theta

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import PIL.Image as Image


    # Load observed image and calculate the gradient
    img_path = files("testing_data") / "im1_kernel1_img.png"
    img = np.array(Image.open(img_path)).astype(float) / 255
    gx, gy = np.gradient(img)
    y = np.stack([gx, gy])

    # Load the true psf
    theta_path = files("testing_data") / "kernel1.png"
    theta_true = np.array(Image.open(theta_path)).astype(float) / 255
    m0 = theta_true.shape[0] // 2

    # Run blind deconvolution
    theta_hat = run(y, m0, lam=1.0)

    # Show the result
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(theta_hat, cmap='gray')
    ax[1].imshow(theta_true, cmap='gray')
    plt.show()

