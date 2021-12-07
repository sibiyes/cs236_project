import os
import sys
import numpy as np

from keras.datasets.fashion_mnist import load_data

import tensorflow as tf; 
#tf.compat.v1.disable_eager_execution()
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Lambda, Reshape
from keras.models import Model
from keras.losses import binary_crossentropy

np.random.seed(25)
tf.executing_eagerly()

def compute_latent(x):
    mu, sigma = x
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=(batch,dim))
    return mu + K.exp(sigma/2)*eps

def kl_reconstruction_loss(true, pred):
    img_width = 28
    img_height = 28
    # Reconstruction loss
    reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height
    # KL divergence loss
    kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    # Total loss = 50% rec + 50% KL divergence loss
    return K.mean(reconstruction_loss + kl_loss)

def main():

    (X_train, Y_train), (X_test, Y_test) = load_data()
    X_train = X_train/255
    X_test = X_test/255
    
    
    print(X_train)
    print(Y_train)
    
    ### filter by category
    X_train = X_train[Y_train == 0, :, :]
    Y_train = Y_train[Y_train == 0]
    
    print(X_train[0])
    
    ### reshape
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    
    print(X_train.shape)
    print(X_test.shape)
    
    input_shape = (28, 28, 1)
    z_dim = 2
    
    ### Encoder ###
    
    encoder_input = Input(shape=input_shape)
    encoder_conv = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(encoder_input)
    encoder_conv = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(encoder_conv)
    encoder = Flatten()(encoder_conv)

    mu = Dense(z_dim)(encoder)
    sigma = Dense(z_dim)(encoder)
    
    ### Latent space
    
    latent_space = Lambda(compute_latent, output_shape=(z_dim,))([mu, sigma])
    
    conv_shape = K.int_shape(encoder_conv)
    print(conv_shape)
    
    # ### Decoder ###
    
    decoder_input = Input(shape=(z_dim,))
    decoder = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation='relu')(decoder_input)
    decoder = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(decoder)
    decoder_conv = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(decoder)
    decoder_conv = Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(decoder_conv)
    decoder_conv =  Conv2DTranspose(filters=1, kernel_size=3, padding='same', activation='sigmoid')(decoder_conv)
    
    ### Defining Model
    
    encoder = Model(encoder_input, latent_space)
    decoder = Model(decoder_input, decoder_conv)
    
    def kl_reconstruction_loss(true, pred):
        img_width = 28
        img_height = 28
        # Reconstruction loss
        reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height
        # KL divergence loss
        kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # Total loss = 50% rec + 50% KL divergence loss
        return K.mean(reconstruction_loss + kl_loss)
    
    vae = Model(encoder_input, decoder(encoder(encoder_input)))
    print(vae)
    vae.compile(optimizer='adam', loss=kl_reconstruction_loss(encoder_input, decoder_conv))

    
    #history = vae.fit(x=X_train_new, y=X_train_new, epochs=20, batch_size=32, validation_data=(X_test_new,X_test_new))
    vae.fit(x=X_train_new, y=X_train_new, epochs=20, batch_size=32, validation_data=(X_test_new,X_test_new))
    
    
if __name__ == '__main__':
    main()    
    
