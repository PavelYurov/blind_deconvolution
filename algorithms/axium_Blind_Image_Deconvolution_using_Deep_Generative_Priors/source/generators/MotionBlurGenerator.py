import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import initializers
from keras.layers import (Conv2D, Conv2DTranspose, Dense, Flatten, Input,
                          Lambda, MaxPooling2D, Reshape, UpSampling2D)
from keras.models import Model


class MotionBlur:
    def __init__(self):
        self.latent_dim = 50  # Dimension of Latent Representation
        self.Encode = None
        self.Decoder = None
        self.VAE = None
        self.weights_path = './model weights/motionblur.h5'

    def GenerateModel(self):
        # Encoder
        input_ = Input(shape=(28, 28, 1))
        encoder_hidden1 = Conv2D(filters=20, kernel_size=2, strides=(1, 1), padding='valid', activation='relu')(input_)
        encoder_hidden2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_hidden1)
        encoder_hidden3 = Conv2D(filters=20, kernel_size=2, strides=(1, 1), padding='valid', activation='relu')(encoder_hidden2)
        encoder_hidden4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(encoder_hidden3)
        encoder_hidden5 = Flatten()(encoder_hidden4)

        latent_units = int(encoder_hidden5.shape[-1])
        spatial_shape = tuple(int(dim) for dim in encoder_hidden4.shape[1:])

        # Latent Representation Distribution, P(z)
        z_mean = Dense(self.latent_dim, activation='linear', kernel_initializer=initializers.he_normal())(encoder_hidden5)
        z_std_sq_log = Dense(self.latent_dim, activation='linear', kernel_initializer=initializers.he_normal())(encoder_hidden5)

        def blur_sample_z(args):
            mu, std_sq_log = args
            epsilon = tf.random.normal((tf.shape(mu)[0], self.latent_dim), dtype=mu.dtype)
            std = tf.exp(0.5 * std_sq_log)
            return mu + epsilon * std

        z = Lambda(blur_sample_z, output_shape=(self.latent_dim,))([z_mean, z_std_sq_log])

        # Decoder/Generator
        decoded_hidden1 = Dense(latent_units, activation='relu', kernel_initializer=initializers.he_normal())(z)
        decoded_hidden2 = Reshape(spatial_shape)(decoded_hidden1)
        decoder_hidden3 = UpSampling2D(size=(2, 2))(decoded_hidden2)
        decoder_hidden4 = Conv2DTranspose(filters=20, kernel_size=2, strides=(1, 1), padding='valid', activation='relu')(decoder_hidden3)
        decoder_hidden5 = UpSampling2D(size=(2, 2))(decoder_hidden4)
        decoder_hidden6 = Conv2DTranspose(filters=20, kernel_size=2, strides=(1, 1), padding='valid', activation='relu')(decoder_hidden5)
        output = Conv2DTranspose(filters=1, kernel_size=2, strides=(1, 1), padding='valid', activation='relu')(decoder_hidden6)

        self.VAE = Model(input_, output)
        self.Encoder = Model(inputs=input_, outputs=[z_mean, z_std_sq_log])

        blur_input_decoder = Input(shape=(self.latent_dim,))
        blur_hidden1_decoder = self.VAE.layers[9](blur_input_decoder)
        blur_hidden2_decoder = self.VAE.layers[10](blur_hidden1_decoder)
        blur_hidden3_decoder = self.VAE.layers[11](blur_hidden2_decoder)
        blur_hidden4_decoder = self.VAE.layers[12](blur_hidden3_decoder)
        blur_hidden5_decoder = self.VAE.layers[13](blur_hidden4_decoder)
        blur_hidden6_decoder = self.VAE.layers[14](blur_hidden5_decoder)
        blur_output_decoder = self.VAE.layers[15](blur_hidden6_decoder)

        self.Decoder = Model(blur_input_decoder, blur_output_decoder)

    def LoadWeights(self):
        self.VAE.load_weights(self.weights_path)

    def GetModels(self):
        return self.VAE, self.Encoder, self.Decoder


if __name__ == '__main__':
    BLURGen = MotionBlur()
    BLURGen.GenerateModel()
    BLURGen.weights_path = '../model weights/motionblur.h5'
    BLURGen.LoadWeights()
    blur_vae, blur_encoder, blur_decoder = BLURGen.GetModels()

    n_samples = 10
    len_z = BLURGen.latent_dim
    z = np.random.normal(0, 1, size=(n_samples * n_samples, len_z))
    sampled = blur_decoder.predict(z)

    k = 0
    for i in range(n_samples):
        for j in range(n_samples):
            blur = sampled[k]
            blur = blur / blur.max()
            blur = blur[:, :, 0]
            plt.subplot(n_samples, n_samples, k + 1)
            plt.imshow(blur, cmap='gray')
            plt.axis('Off')
            k = k + 1
    plt.show()
