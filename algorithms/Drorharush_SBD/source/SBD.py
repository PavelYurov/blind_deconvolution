import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pylops as pl
import random

import scipy.stats
from scipy import sparse
from autograd.scipy.signal import convolve as aconvolve
from scipy.signal import convolve
from numpy.random import default_rng
import autograd.numpy as anp
from numpy.linalg import norm
from pymanopt.manifolds import Sphere
from pymanopt import Problem
from pymanopt.optimizers import TrustRegions
from pymanopt.function import autograd as ad
from .model import ActivationNet, LISTA
from pathlib import Path
import torch
import os
from scipy import io
from tqdm import tqdm
from nanonispy.read import Grid
from dataclasses import dataclass
from abc import ABC, abstractmethod
from dataclasses import field


@dataclass
class Measurement:
    density_of_states: np.ndarray = None
    kernel: np.ndarray = None
    kernel_size: (int, int) = None
    activation_map = None
    topography: np.ndarray = None


@dataclass
class DataHandler(ABC):
    """Represents a generic data handler."""
    @abstractmethod
    def get_measurement_data(self) -> Measurement:
        """Returns a measurement."""
        pass


@dataclass
class ThreeDSHandler(DataHandler):
    """DataHandler that handles .3ds files"""
    file_path: str
    kernel_size: (int, int)
    grid: Grid = field(init=False)

    def __post_init__(self):
        self.grid = Grid(self.file_path)

    def get_measurement_data(self, normalize=False, index_list=None) -> Measurement:
        """Returns the file data as a Measurement"""
        # If you chose to normalize the DoS using the current map
        if normalize:
            density_of_states = np.moveaxis(
                np.divide(self.grid.signals['LIX 1 omega (A)'], self.grid.signals['Current (A)']), -1, 0)
        else:
            density_of_states = np.moveaxis(self.grid.signals['LIX 1 omega (A)'], -1, 0)

        # If you only want part of the energy levels
        if index_list is not None:
            # Making sure all the given indexes exist in the DoS map
            assert np.all(np.isin(index_list, [
                *range(np.shape(density_of_states)[0])])), 'Not all the indeces in the list provided exist.'
            density_of_states = [density_of_states[index] for index in index_list]
        return Measurement(
            density_of_states=density_of_states,
            topography=self.grid.signals['topo'],
            kernel_size=self.kernel_size)


@dataclass
class Simulation:
    """Runs and stores simulated measurements."""
    levels: int
    measurement_size: (int, int)
    kernel_size: (int, int)
    defect_density: float = 0.01
    SNR: float = 1
    lattice_structure: str = None
    index_list: list = None
    measurement: Measurement = None

    def run(self) -> None:
        """Runs the simulation again. Over-writes the measurement instance variable."""
        density_of_states, kernel, activation_map = Y_factory(
            s=self.levels,
            Y_size=self.measurement_size,
            A_size=self.kernel_size,
            density=self.defect_density,
            SNR=self.SNR,
            lattice_structure=self.lattice_structure)
        self.measurement = Measurement(density_of_states, kernel, self.kernel_size, activation_map)


def deconvolve(measurement: Measurement, use_topo=False) -> tuple:
    """Takes a measurement, deconvolves it and returns (Kernel, Activation map) tuple"""
    X_solved = measurement_to_activation(measurement, use_topo=use_topo)
    A_solved = measurement_to_ker(measurement, X_solved)

    return A_solved, X_solved


def deconv_v0(Y, kernel_size, l_i, l_f, alpha):
    """
    Preforms blind de-convolution given a measurement, kernel size, initial and final lambdas (sparsity), and decay rate.
    :param Y: measurement with shape (s, n1, n2)
    :param kernel_size: size of single real-space QPI, with shape (s, m1, m2)
    :param l_i: Initial sparsity constant. Bigger than l_f.
    :param l_f: Final sparsity constant. Bigger than l_f.
    :param alpha: Decay rate from l_i to l_f. Between zero and one.
    :return: returns the kernel and activation maps in the shapes of Y and kernel_size respectively.
    """

    assert l_i >= l_f, "l_i is smaller than l_f."

    # Initialization

    A = np.random.uniform(low=np.amin(Y), high=np.amax(Y), size=kernel_size)
    A, X = Asolve(A, l_i, Y)

    # Refinement

    kernel_array = np.array(kernel_size)
    pad = np.floor(kernel_array[1:] / 2)
    A_big = np.pad(A, ((0, 0), (pad[0], pad[0]), (pad[1], pad[1])))
    sparse = l_i
    iteration_number = np.ceil(np.log(l_f / l_i) / np.log(alpha))

    for _ in range(iteration_number):
        A_big, X = Asolve(A_big, sparse, Y, X)
        A_big, X = center(kernel_array, A, X)
        sparse = alpha * sparse

    # Output

    A_out = crop_to_center(A_big, kernel_array)
    X_out = FISTA(sparse, A_out, X)
    return A_out, X_out


def measurement_to_ker(measurement: Measurement, activation_map) -> np.ndarray:
    """Takes a measurement and an activation map and returns the recovered kernel."""
    # Initial random guess
    kernel_rnd = np.random.normal(0, 1, tuple(2 * m for m in measurement.kernel_size))
    kernel_rnd = kernel_rnd / np.linalg.norm(kernel_rnd)  # Norm = 1
    level_num, _, _ = np.shape(measurement.density_of_states)

    # Solving
    kernel = np.zeros((level_num,) + tuple(2 * m for m in measurement.kernel_size))
    dos = measurement.density_of_states
    for i in range(level_num):
        # Taking the random guess for the 1st level
        if i == 0:
            first_guess = kernel_rnd
        else:
            first_guess = kernel[i - 1]
        kernel[i] = np.array(RTRM(1e-5, activation_map, normalize_measurement(dos[i]), first_guess)[0])

    # Cropping and normalizing
    kernel = crop_to_center(kernel, measurement.kernel_size)
    return sphere_norm_by_layer(kernel)


def measurement_to_activation(measurement: Measurement, model='lista', use_topo=False) -> np.ndarray:
    """
    Takes in a measurement object (and optionally use_topo flag) and returns the recovered activation map
    """
    dos = measurement.topography if use_topo else measurement.density_of_states
    assert np.ndim(dos) in [2, 3], f'Your measurement has {np.ndim(dos)} dimensions'
    # Loading the selected network
    if model == 'cnn':
        net = ActivationNet()
        trained_model_path = Path('trained_model_norm.pt', map_location=torch.device('cpu'))
    elif model == 'lista':
        net = LISTA(5, iter_num=10)
        trained_model_path = Path('trained_lista_5layers.pt', map_location=torch.device('cpu'))
    if trained_model_path.is_file():
        net.load_state_dict(torch.load(trained_model_path))
        print('Loaded parameters from your trained model.')
    else:
        print('No trained model detected.')
    net.eval()
    net.cpu()
    net.double()
    mes_shape = np.shape(dos)

    # If there are layers, it will calculate the mean activation map
    if np.ndim(dos) == 3:
        activations = np.zeros(mes_shape)

        for level in range(mes_shape[0]):
            temp_mes = dos[level]
            temp_mes = temp_mes[np.newaxis, :, :]
            measurement_tensor = ndarray_to_tensor(temp_mes)
            temp_act = net(measurement_tensor)[0][0].data.numpy()

            # This next bit is a filter to zero out all the small numbers in the map
            # temp_act[temp_act < np.max(temp_act) / 100] = 0
            temp_act = temp_act / np.sum(temp_act)
            activations[level] = temp_act

        activation = combine_activation_maps(activations)

    else:
        measurement_tensor = ndarray_to_tensor(np.expand_dims(dos, 0))
        activation = net(measurement_tensor)[0][0].data.numpy()

    # Threshold: points under 100th of the max are set to zero
    activation[activation < np.max(activation) / 100] = 0
    activation = activation / np.sum(activation)
    # activation = net(ndarray_to_tensor(np.expand_dims(activation, 0)))[0][0].data.numpy()
    return activation


def combine_activation_maps(activation_maps: np.ndarray) -> np.ndarray:
    """This function takes a stack of activation maps and returns the masked median"""
    assert np.ndim(
        activation_maps) == 3, f'The activation maps array has to have 3 dims, not {np.ndim(activation_maps)}'
    level_num = np.shape(activation_maps)[0]
    bool_maps = np.ones(np.shape(activation_maps), dtype=bool)

    # Mask preparation
    bool_maps = np.array([activ_map > np.max(activ_map) / 100 for activ_map in activation_maps])
    
    mask = scipy.stats.mode(bool_maps.astype(int), axis=0)[0][0]

    activation = np.median(activation_maps, axis=0)
    activation = activation * mask
    total = np.sum(activation)
    if not np.isfinite(total) or total <= 0:
        return np.zeros_like(activation)
    activation /= total  # Normalize so sum will be 1
    return activation


def RTRM(lam_in, X_in, Y, A0, min_grad_norm=5e-7, max_time=30.0, verbosity=0):
    """
    Minimizing A for cost_fun using Riemannian Trust-Region Method (RTRM) over the sphere.
    """
    A_in = anp.expand_dims(A0, 0) if anp.ndim(A0) == 2 else A0
    Y_in = anp.expand_dims(Y, 0) if anp.ndim(A0) == 2 else Y
    s, m1, m2 = anp.shape(A_in)
    sphere = Sphere(s, m1, m2)  # Sphere принимает произвольный shape

    @ad(sphere)               # <-- важное отличие: передаём manifold в декоратор
    def cost(A):
        # внутри cost используем autograd.numpy (anp)!
        return cost_fun(lam_in, A, X_in, Y_in)

    problem = Problem(manifold=sphere, cost=cost)

    solver = TrustRegions(
        min_gradient_norm=min_grad_norm,
        max_time=max_time,
        verbosity=verbosity,
        log_verbosity=verbosity  # если хочешь лог по итерациям, оставь 1
    )

    result = solver.run(problem, initial_point=A_in)
    return result.point


def FISTA(lam_in, A_in, Y, niter=1):
    s, n1, n2 = np.shape(Y)
    s, m1, m2 = np.shape(A_in)

    Cop = pl.signalprocessing.ConvolveND(N=s * n1 * n2, h=A_in, dims=(s, n1, n2), offset=(0, m1 // 2, m2 // 2),
                                         dirs=[1, 2])
    X = pl.optimization.sparsity.FISTA(Cop, Y.flatten(), niter, eps=2 * lam_in, threshkind='hard')[0]

    X = X.reshape((s, n1, n2))
    return np.median(X, axis=0)


def cost_fun(lambda_in, A, X, Y):
    """
    0.5 * ||A ⨂ X - Y||^2 + lambda * r(X)
    A: (s, m1, m2), X: (...), Y: (s, n1, n2)
    """
    # убеждаемся, что это autograd-массивы
    A = anp.array(A)
    X = anp.array(X)
    Y = anp.array(Y)

    s, n1, n2 = Y.shape

    # свёртка (оставляем вашу aconvolve)
    A_con_X = aconvolve(X, A, mode='full', axes=([0, 1], [1, 2]))
    A_con_X = crop_to_center(A_con_X, (n1, n2))

    data_term = 0.5 * anp.sum((A_con_X - Y) ** 2)

    # регуляризатор (пример: L1 на X; если у вас другой r, вставьте его тут)
    reg_term = lambda_in * anp.sum(anp.abs(X))

    return data_term + reg_term


def regulator(X):
    """
    The pseudo-Huber regulator
    :param X: 2D matrix
    :return: A real number
    """
    mu = 10 ** -6  # A small positive number (chosen in the paper to be 10 ** -6)
    return np.sum(mu ** 2 * (anp.sqrt(1 + (mu ** -2) * X) - 1))


def Asolve(A_in, lambda_in, Y, X=None):
    X_in = FISTA(lambda_in, A_in, Y) if X is None else X
    A_out = RTRM(lambda_in, X_in, Y, A_in)
    X_out = FISTA(lambda_in, A_in, Y)
    raise NotImplementedError


def ndarray_to_tensor(array):
    """
    Turns an ndarray to a torch tensor
    """
    return torch.tensor(array.astype(float)).unsqueeze(dim=0)


def Y_factory(s, Y_size, A_size, density, SNR: float = 1, lattice_structure=None):
    """
    This function produces a QPI measurement with specified size, defect density, kernel size and number of levels
    """
    n1, n2 = Y_size
    m1, m2 = A_size
    A = kernel_factory(s, m1, m2)
    X = sparse.random(n1, n2, density)
    X = X / np.sum(X)

    signal = np.array([convolve(X.A, A[level], mode='same') for level in range(s)])
    signal_var = np.var(signal, axis=(-2, -1))
    noise_var = signal_var / SNR
    noise = np.random.normal(0, 1, size=(s, n1, n2)) * np.sqrt(noise_var)[:, None, None]

    if lattice_structure is None:
        Y = signal + noise
    else:
        lattice = lattice_background(s, Y_size, A_size, lattice_structure)
        lattice *= np.sqrt(noise_var / np.var(lattice))[:, None, None]  # Makes lattice var the same as noise_var
        Y = signal + (noise + lattice) / np.sqrt(2)  # Makes the total non-signal var: signal_var / SNR
    return Y, A, X


def kernel_factory(s: int, m1: int, m2: int):
    """
    This function produces a set of s random m1 by m2 kernels
    """
    m_max = max(m1, m2)
    A = np.zeros([s, m_max, m_max], dtype=float)
    symmetry = random.choice([2, 3, 4, 6])
    half_sym = np.floor(symmetry / 2).astype('int')
    lowest_k = 0.5
    highest_k = 3
    k = np.random.uniform(lowest_k, highest_k, [s, symmetry])
    x, y = np.meshgrid(np.linspace(-1, 1, m_max), np.linspace(-1, 1, m_max))
    arb_angle = np.random.uniform(0, 2 * np.pi)

    for direction in range(symmetry):
        ang = direction * 180 / symmetry
        ang = arb_angle + ang * np.pi / 180
        r = (x * np.cos(ang) + np.sin(ang) * y)
        phi = np.random.uniform(0, 2 * np.pi)
        for i in range(s):
            # Adding normal decay
            sigma = np.random.uniform(0.2, 0.5)
            decay = gaussian_window(m_max, m_max, sigma)
            A[i, :, :] += np.cos(2 * np.pi * k[i, direction % half_sym] * r) * decay

    # Normalizing:
    # A = np.abs(A)
    A = sphere_norm_by_layer(A)
    return A


def lattice_background(s: int, measurement_size: (int, int), kernel_size: (int, int),
                       structure: str = None) -> np.ndarray:
    """
    Generates a square lattice background in a given size
    """
    m_max = np.max(kernel_size)
    aspect_ratio = measurement_size[0] / measurement_size[1]
    atoms_in_kernel = np.random.uniform(0.9, 3)
    lambda_lattice = (np.max(kernel_size) / np.max(measurement_size)) / (2 * atoms_in_kernel)
    background = np.zeros((s, measurement_size[0], measurement_size[1]))
    x, y = np.meshgrid(np.linspace(-1, 1, measurement_size[1]),
                       np.linspace(-aspect_ratio, aspect_ratio, measurement_size[0]))  # Add nm scale instead of [-1,1]?
    theta = np.arctan(y / x)
    r = np.sqrt(x ** 2 + y ** 2)
    arb_angle = np.random.uniform(0, np.pi)
    d_theta = theta - arb_angle
    if structure is None:
        structure = random.choice(['square', 'honeycomb'])
    if structure == 'square':
        dir1, dir2 = r * np.cos(d_theta), r * np.sin(d_theta)
        k1, k2 = 2 * np.pi / lambda_lattice, 2 * np.pi / lambda_lattice
        background = np.array([np.cos(dir1 * k1) + np.cos(dir2 * k2) for _ in range(s)])
    elif structure == 'honeycomb':
        dir1, dir2, dir3 = r * np.cos(d_theta), r * np.cos(2 * np.pi / 3 - d_theta), r * np.cos(2 * np.pi / 3 + d_theta)
        k1, k2, k3 = 2 * np.pi / lambda_lattice, 2 * np.pi / lambda_lattice, 2 * np.pi / lambda_lattice
        background = np.array([np.cos(dir1 * k1) + np.cos(dir2 * k2) + np.cos(dir3 * k3) for _ in range(s)])
    background = sphere_norm_by_layer(background)
    return background


def sphere_norm_by_layer(M):
    """
    Returns your matrix with each layer normalized to the unit sphere.
    """
    assert len(np.shape(M)) == 3, 'The matrix does not have 3 dim'
    return M / norm(M, axis=(-2, -1))[:, None, None]


def gaussian_window(n1, n2, sig=1, mu=0):
    """
    This function produces a 2D gaussian of size n1 by n2
    """
    x, y = np.meshgrid(np.linspace(-1, 1, n1), np.linspace(-1, 1, n2))
    d = np.sqrt(x * x + y * y)
    return np.exp(-((d - mu) ** 2 / (2.0 * sig ** 2)))


def center(size, A, X):
    """
    This function shifts the kernel (A) to center around its max sub-matrix, and shifts the activation map (X) to the
    opposite direction.
    """
    n1, n2 = np.shape(A)[-2], np.shape(A)[-1]
    s1, s2 = size[-2], size[-1]
    center_corner1 = np.floor((n1 - s1) / 2)
    center_corner2 = np.floor((n2 - s2) / 2)
    max_corner = max_submatrix_pos(A, size[-2], size[-1])
    A_shift = tuple([np.floor(center_corner1 - max_corner[0]).astype('int'),
                     np.floor(center_corner2 - max_corner[1]).astype('int')])
    X_shift = tuple(- j for j in A_shift)
    centered_A = shift(A, A_shift)
    centered_X = shift(X, X_shift)

    return centered_A, centered_X


def shift(mt, s=(0, 0)):
    """
    Takes a matrix and a tuple and returns that matrix shifted in the last 2 axes according to the tuple.
    :param mt: matrix, shape len 2 or 3
    :param s: tuple defining the shift (row shift, column shift)
    :return: shifted matrix
    """
    z = np.zeros(np.shape(mt))
    s1, s2 = s
    row_shift = list(range(-s1, -s1 + z.shape[-2]))
    col_shift = list(range(-s2, -s2 + z.shape[-1]))
    for row in range(z.shape[-2]):
        for col in range(z.shape[-1]):
            in_bounds = z.shape[1] > row_shift[row] >= 0 and z.shape[2] > col_shift[col] >= 0
            if len(z.shape) == 2:
                z[row, col] = mt[row_shift[row], col_shift[col]] if in_bounds else 0
            elif len(z.shape) == 3:
                z[:, row, col] = mt[:, row_shift[row], col_shift[col]] if in_bounds else 0

    return z


def max_submatrix_pos(matrix, m1, m2):
    """
    Returns position of max submatrix.
    :param matrix: Matrix with shape (i, row, col)
    :param m1: rows in kernel
    :param m2: columns in kernel
    :return: row, column of max submatrix of size m1 by m2 in matrix
    """
    # Initializing
    sub_norm = 0
    pos1, pos2 = 0, 0
    arr = np.sum(np.array(matrix), axis=0)
    rows = arr.shape[0]
    cols = arr.shape[1]

    # Iterating
    for row in range(rows - m1 + 1):
        for col in range(cols - m2 + 1):
            running_sum = np.sum(arr[row:row + m1, col:col + m2])
            if running_sum > sub_norm:
                sub_norm = running_sum
                pos1, pos2 = row, col
    return np.array([pos1, pos2])


def crop_to_center(M, box_shape):
    """
    Crops a (s, n1, n2) matrix to a (s, m1, m2) matrix around the center
    :param M: dim=3 matrix
    :param box_shape: (m1, m2) or (s, m1, m2)
    :return: dim=3 matrix
    """
    if len(box_shape) == 3:
        diff = np.subtract(np.shape(M)[1:], box_shape[1:])
        gap = diff // 2
        return M[:, gap[0]:gap[0] + box_shape[1], gap[1]:gap[1] + box_shape[2]]
    diff = np.subtract(np.shape(M)[1:], box_shape)
    gap = diff // 2
    return M[:, gap[0]:gap[0] + box_shape[0], gap[1]:gap[1] + box_shape[1]]


def recovery_error(A_guess, A_true):
    """
    Returns the recovery error defined by: (2/pi) * arccos|<A_g, A_t>|
    """
    return (2 / np.pi) * np.arccos(np.round(abs(np.dot(A_guess.flatten(), A_true.flatten())), 10))


def benchmark(model, defect_density_range, kernel_size_range, SNR=2, samples=20):
    """
    Returns a recovery error matrix
    """
    error_matrix = np.zeros([len(defect_density_range), len(kernel_size_range)])
    for i, defect_density in enumerate(tqdm(defect_density_range)):
        print(f'loop {i + 1} out of {len(defect_density_range)} \n')
        for j, kernel_size in enumerate(kernel_size_range):
            errors = np.zeros(samples)
            for sample in range(samples):
                Y, A, X = Y_factory(1, (256, 256), (kernel_size, kernel_size), defect_density, SNR)

                X_guess = measurement_to_activation(Y, model=model)

                A_rand = np.random.normal(0, 1, (1, 2 * kernel_size, 2 * kernel_size))
                A_rand = A_rand / np.linalg.norm(A_rand)

                A_solved = RTRM(1e-5, X_guess, Y, A_rand)
                A_solved = crop_to_center(A_solved, (1, kernel_size, kernel_size))
                A_solved = A_solved / np.linalg.norm(A_solved)  # norm = 1

                errors[sample] = recovery_error(A_solved[0], A[0])
            error_matrix[i, j] = np.median(errors)
    return error_matrix


def compare_fft_plot(matrix1, matrix2, title):
    bigger_shape = np.shape(matrix1) if np.shape(matrix1)[1] > np.shape(matrix2)[1] else np.shape(matrix2)
    fft_matrix1 = np.fft.fftshift(np.fft.fft2(matrix1, s=bigger_shape))
    fft_matrix2 = np.fft.fftshift(np.fft.fft2(matrix2, s=bigger_shape))

    # Real Part
    fig, ax = plt.subplots(2, 3, figsize=(18, 6))
    fig.suptitle(title)

    ax[0, 0].imshow(np.real(fft_matrix1), cmap='bwr', norm=colors.CenteredNorm())
    ax[0, 0].set_title('Real Part 1')
    ax[0, 1].imshow(np.imag(fft_matrix1), cmap='bwr', norm=colors.CenteredNorm())
    ax[0, 1].set_title('Imaginary Part 1')
    ax[0, 2].imshow(np.abs(fft_matrix1), cmap='bwr', norm=colors.CenteredNorm())
    ax[0, 2].set_title('Absolute Value 1')

    ax[1, 0].imshow(np.real(fft_matrix2), cmap='bwr', norm=colors.CenteredNorm())
    ax[1, 0].set_title('Real Part 2')
    ax[1, 1].imshow(np.imag(fft_matrix2), cmap='bwr', norm=colors.CenteredNorm())
    ax[1, 1].set_title('Imaginary Part 2')
    ax[1, 2].imshow(np.abs(fft_matrix2), cmap='bwr', norm=colors.CenteredNorm())
    ax[1, 2].set_title('Absolute Value 2')
    for j in range(2):
        for i in range(3):
            ax[j, i].set_axis_off()
    plt.tight_layout()
    plt.show()


def save_data(number_of_samples, measurement_size, kernel_size, SNR=2, training=False, validation=False, testing=False):
    files_in_folder = os.listdir()
    if training and 'training_dataset' not in files_in_folder:
        os.system("mkdir training_dataset")
    if validation and 'validation_dataset' not in files_in_folder:
        os.system("mkdir validation_dataset")
    if testing and 'testing_dataset' not in files_in_folder:
        os.system("mkdir testing_dataset")

    density_exponent = np.random.uniform(low=-3.5, high=-1.5, size=(number_of_samples,))

    E, n1, n2 = measurement_size
    # sample_ker_size = np.random.randint(low=kernel_size[0] / 2, high=kernel_size[0] * 2, size=number_of_samples)
    sample_SNR = np.random.uniform(SNR / 2, 5 * SNR, number_of_samples)

    for i in range(number_of_samples):
        temp_measurement, temp_kernel, temp_activation_map = Y_factory(E, (n1, n2),
                                                                       kernel_size,
                                                                       10 ** density_exponent[i],
                                                                       sample_SNR[i])
        if training:
            np.save(os.getcwd() + '/training_dataset/kernel_%d' % i, temp_kernel)
            np.save(os.getcwd() + '/training_dataset/measurement_%d' % i, temp_measurement)
            io.mmwrite(os.getcwd() + '/training_dataset/activation_%d' % i, temp_activation_map)
        elif validation:
            np.save(os.getcwd() + '/validation_dataset/kernel_%d' % i, temp_kernel)
            np.save(os.getcwd() + '/validation_dataset/measurement_%d' % i, temp_measurement)
            io.mmwrite(os.getcwd() + '/validation_dataset/activation_%d' % i, temp_activation_map)
        elif testing:
            np.save(os.getcwd() + '/testing_dataset/kernel_%d' % i, temp_kernel)
            np.save(os.getcwd() + '/testing_dataset/measurement_%d' % i, temp_measurement)
            io.mmwrite(os.getcwd() + '/testing_dataset/activation_%d' % i, temp_activation_map)
    if not training and not validation and not testing:
        print("Specify validation or training to save files.")


def normalize_measurement(Y):
    """
    Normalizes the input such that mean[Y]=0 and STD[Y]=0.0025
    """
    return (0.0025 / np.std(Y)) * (Y - np.mean(Y))


def plane_fit(matrix: np.ndarray, plot: bool = False) -> np.ndarray:
    m_max, m_min = np.max(matrix), np.min(matrix)
    m1, m2 = matrix.shape
    x1, x2 = np.mgrid[:m1, :m2]

    # Normalizing:
    norm_matrix = (matrix - m_min)/(m_max - m_min)

    X = np.hstack((np.reshape(x1, (m1 * m2, 1)), np.reshape(x2, (m1 * m2, 1))))
    X = np.hstack((np.ones((m1 * m2, 1)), X))
    YY = np.reshape(norm_matrix, (m1 * m2, 1))

    # Regression through the normal equation
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)

    plane = np.reshape(np.dot(X, theta), (m1, m2))

    # Plotting:
    if plot:
        plot_plane(x1, x2, norm_matrix, plane)
    return plane


def plot_plane(x1, x2, norm_matrix, plane):
    fig = plt.figure()
    ax = fig.add_subplot(3, 1, 1, projection='3d')
    jet = plt.get_cmap('jet')
    ax.plot_surface(x1, x2, norm_matrix, rstride=1, cstride=1, cmap=jet, linewidth=0)
    ax = fig.add_subplot(3, 1, 2, projection='3d')
    ax.plot_surface(x1, x2, plane)
    ax.plot_surface(x1, x2, norm_matrix, rstride=1, cstride=1, cmap=jet, linewidth=0)

    Y_sub = norm_matrix - plane
    ax = fig.add_subplot(3, 1, 3, projection='3d')
    ax.plot_surface(x1, x2, Y_sub, rstride=1, cstride=1, cmap=jet, linewidth=0)
    plt.show()


if __name__ == '__main__':
    measurement_shape = (1, 128, 128)
    kernel_shape = (16, 16)

    save_data(10000, measurement_shape, kernel_shape, training=True)
    save_data(2000, measurement_shape, kernel_shape, validation=True)
    save_data(100, (1, 200, 200), (20, 20), testing=True)
