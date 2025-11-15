import numpy as np

def dot(theta, gamma, lam, x):
    """
    Computes (lam*A(theta)^T*A(theta) + Gamma)x
    :param theta: point-spread function (M_1, M_2)
    :param gamma: variational parameters (..., N_1, N_2)
    :param lam: measurement noise precision
    :param x: (..., N_1, N_2)
    :return: The matrix-vector product (lam*A(theta)^T*A(theta) + Gamma)x
    """

    # FFT convolution
    Theta = np.fft.fft2(theta)
    X = np.fft.fft2(x)
    x_conv = np.real(np.fft.ifft2(X * np.conj(Theta)*Theta))

    return lam*x_conv + gamma*x


def conjugate_grad(theta, gamma, lam, b, x=None):
    """
    Conjugate gradient method for solving (lam*A(theta)^T*A(theta) + Gamma)x = b
    :param theta: point-spread function (M_1, M_2)
    :param gamma: variational parameters (..., N_1, N_2)
    :param lam: measurement noise precision
    :param b: right-hand side (..., N_1, N_2)
    :param x: (..., N_1, N_2)
    """
    n = b.size
    if not x:
        x = np.ones_like(b)
    r = dot(theta, gamma, lam, x) - b
    p = - r
    r_k_norm = np.sum(r * r, axis=(-2, -1), keepdims=True)
    for i in range(2*n):
        Ap = dot(theta, gamma, lam, p)
        alpha = r_k_norm / np.sum(p * Ap, axis=(-2, -1), keepdims=True)
        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = np.sum(r * r, axis=(-2, -1), keepdims=True)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if all(r_kplus1_norm < 1e-5):
            print('Itr:', i)
            break
        p = beta * p - r

    return x

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import PIL.Image as Image
    from importlib.resources import files

    from three_implementations_of_convolution import kernel_to_ccb, gkern

    img_path = files("testing_data") / "im1.png"
    x = np.array(Image.open(img_path)).astype(float) / 255
    x = x[:100, 100:200]
    gx, gy = np.gradient(x)
    x = np.stack([gx, gy])
    channels, height, width = x.shape
    theta = gkern(11, 2.0)
    gamma = np.ones_like(x)

    # Matrix-vector convolution
    hpad = x.shape[-2] - theta.shape[0]
    vpad = x.shape[-1] - theta.shape[1]
    theta = np.pad(theta, [[0, hpad], [0, vpad]])
    theta = np.roll(theta, -5, axis=0)
    theta = np.roll(theta, -5, axis=1)

    lam = 10
    x_vec = x.reshape(channels, height*width)
    gamma_vec = gamma.reshape(channels, height*width)
    A = kernel_to_ccb(theta)
    H = lam * (A.T @ A)[None, :, :] + np.stack([np.diag(gi) for gi in gamma_vec])
    b_vec = H @ x_vec[:, :, None]

    x_hat_vec = np.linalg.solve(H, b_vec)
    x_hat = conjugate_grad(theta, gamma, lam, b_vec.reshape(channels, height, width))

    print("Error: ", np.sum(np.abs(x_hat_vec.reshape(channels, height, width) - x_hat)))

    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(x_hat[0, :, :], cmap='gray')
    ax[0,1].imshow(x_hat[1, :, :], cmap='gray')
    ax[1,0].imshow(x_hat_vec.reshape(channels, height, width)[0, :, :], cmap='gray')
    ax[1,1].imshow(x_hat_vec.reshape(channels, height, width)[1, :, :], cmap='gray')
    plt.show()
