import numpy as np

def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def circ(b):
    width = len(b)
    P = np.eye(width)[:, [(i+1) % width for i in range(width)]]
    B = b[None].T
    for i in range(width-1):
        b = P @ b
        B = np.hstack((B, b[None].T))
    return B

def block_circ(B):
    width = B.shape[1]
    height = B.shape[0] // width

    P = np.eye(height)[:, [(i+1) % height for i in range(height)]]
    P = np.kron(P, np.eye(width))
    C = B
    for i in range(height-1):
        B = P @ B
        C = np.hstack((C, B))

    return C

def kernel_to_ccb(kernel):
    height, width = kernel.shape
    B = circ(kernel[0, :])
    for i in range(height-1):
        B = np.vstack([B, circ(kernel[i+1])])
    return block_circ(B)


if __name__ == '__main__':
    import PIL.Image as Image
    from importlib.resources import files
    import matplotlib.pyplot as plt
    from scipy.signal import convolve2d

    # Load image
    img_path = files("testing_data") / "im1.png"
    img = np.array(Image.open(img_path)).astype(float) / 255
    img = img[:40, :50]
    height, width = img.shape
    kernel = gkern(11, 3.0)

    # Standard convolution
    img_conv = convolve2d(img, kernel, boundary='wrap', mode='same')

    # Matrix-vector convolution
    hpad = img.shape[0] - kernel.shape[0]
    vpad = img.shape[1] - kernel.shape[1]
    kernel = np.pad(kernel, [[0, hpad], [0, vpad]])
    kernel = np.roll(kernel, -5, axis=0)
    kernel = np.roll(kernel, -5, axis=1)

    C = kernel_to_ccb(kernel)
    v = img.ravel()
    v = C @ v
    #v = C.T @ C @ v
    print("Diag: ", np.diag(C.T @ C))
    print(np.sum(kernel*kernel))
    img_conv1 = v.reshape(height, width)

    # FFT convolution
    Kernel = np.fft.fft2(kernel)
    Img = np.fft.fft2(img)
    img_conv2 = np.real(np.fft.ifft2(Img * Kernel))
    #img_conv2 = np.real(np.fft.ifft2(Img * np.conj(Kernel)*Kernel))

    fig, ax = plt.subplots(2, 2)
    ax[0,0].imshow(img_conv, cmap='gray')
    ax[0,1].imshow(img_conv1, cmap='gray')
    ax[1,0].imshow(img_conv2, cmap='gray')
    ax[1,1].imshow(np.abs(img_conv1 - img_conv2), cmap='gray')
    plt.show()