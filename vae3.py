### https://keras.io/examples/generative/vae/
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

import matplotlib.pyplot as plt

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

from vae3_modules import vae_mnist_simple, fashion_product_simple, fashion_product2, fashion_product3, fashion_product4, fashion_product5

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
### https://stackoverflow.com/questions/54323960/save-keras-model-at-specific-epochs
### https://towardsdatascience.com/building-custom-callbacks-with-keras-and-tensorflow-2-85e1b79915a3
class CustomSaver(keras.callbacks.Callback):
    def __init__(self, save_epochs, save_folder):
        super(CustomSaver, self).__init__()
        self.save_epochs = save_epochs
        self.save_folder = save_folder
        
    def on_epoch_end(self, epoch, logs={}):
        #print('epoch:', epoch)
        
        if (epoch % self.save_epochs) == 0:  # or save after some epoch, each k-th epoch etc.
            print('saving checkpoint (epoch {0}) ...'.format(epoch))
            self.model.encoder.save(self.save_folder +  "/model_{}_encoder.h5".format(epoch))
            self.model.decoder.save(self.save_folder + "/model_{}_decoder.h5".format(epoch))
            

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            #print('reconstruction')
            #print(reconstruction)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            #print('kl loss', kl_loss)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            #print('kl loss', kl_loss)
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        
# def train_step(self, data):
#     with tf.GradientTape() as tape:
#         z_mean, z_log_var, z = self.encoder(data)
#         reconstruction = self.decoder(z)
#         print('reconstruction')
#         print(reconstruction)
#         reconstruction_loss = tf.reduce_mean(
#             tf.reduce_sum(
#                 keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
#             )
#         )
#         kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
#         kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
#         total_loss = reconstruction_loss + kl_loss
#     grads = tape.gradient(total_loss, self.trainable_weights)
#     self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
#     self.total_loss_tracker.update_state(total_loss)
#     self.reconstruction_loss_tracker.update_state(reconstruction_loss)
#     self.kl_loss_tracker.update_state(kl_loss)
#     return {
#         "loss": self.total_loss_tracker.result(),
#         "reconstruction_loss": self.reconstruction_loss_tracker.result(),
#         "kl_loss": self.kl_loss_tracker.result(),
#     }
        
def plot_latent_space(vae, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()
    
def plot_label_clusters(z_mean, labels):
    # display a 2D plot of the digit classes in the latent space
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
        
def main():
    
    ### Encoder ###
    latent_dim = 10
    run = 4

    #data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder/apparel/men'
    #data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder_clean/apparel/men'
    data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder_clean/apparel/'
    
    #data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder/test'
    data_set = keras.preprocessing.image_dataset_from_directory(data_folder, label_mode = None, image_size=(128, 128))
    data_set = data_set.map(lambda x: (tf.divide(x, 255)))
    
    encoder, decoder = fashion_product_simple(latent_dim)
    model_tag = 'model_fashion_simple'
    
    model_folder = base_folder + '/output/vae3/models/{0}_z{1}/run{2}'.format(model_tag, latent_dim, run)
    latent_z_folder = base_folder + '/output/vae3/latent_z/{0}_z{1}/run{2}'.format(model_tag, latent_dim, run)
    create_folder(model_folder)
    create_folder(latent_z_folder)
    
    
    ### MNIST ###
    
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # data_set = np.concatenate([x_train, x_test], axis=0)
    # data_set = np.expand_dims(data_set, -1).astype("float32") / 255
    # 
    # encoder, decoder = vae_mnist_simple(latent_dim)
    # 
    # model_folder = base_folder + '/output/vae3/models/model_simple_z{0}/run{1}'.format(latent_dim, run)
    # latent_z_folder = base_folder + '/output/vae3/latent_z/model_simple_z{0}/run{1}'.format(latent_dim, run)
    # create_folder(model_folder)
    # create_folder(latent_z_folder)
    
    ###### Train #######
    
    epochs = 100
    
    # model_folder = base_folder + '/output/vae3/models/model_simple_z{0}/run{1}'.format(latent_dim, run)
    # latent_z_folder = base_folder + '/output/vae3/latent_z/model_simple_z{0}/run{1}'.format(latent_dim, run)
    # create_folder(model_folder)
    # create_folder(latent_z_folder)
                                
    model_save_callback = CustomSaver(save_folder = model_folder, save_epochs = 25)

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
    #vae.fit(mnist_digits, epochs = epochs, batch_size = 128, callbacks = [model_save_callback])
    vae.fit(data_set, epochs = epochs, batch_size = 32, callbacks = [model_save_callback])
    
    vae.encoder.save(model_folder +  "/model_final_encoder.h5")
    vae.decoder.save(model_folder + "/model_final_decoder.h5")
    
    sys.exit(0)
    
    
    #(x_train, y_train), _ = keras.datasets.mnist.load_data()
    (train_images, _), (test_images, _) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255

    ### compute latent variables
    z_mean, _, _ = vae.encoder.predict(x_train)
    #z_mean = pd.DataFrame(z_mean, columns = ['z{0}'.format(i+1) for i in range(latent_dim)])
    print(z_mean)
    np.savetxt(latent_z_folder + '/' + 'latent_z.csv', z_mean, delimiter=",")

    #plot_label_clusters(vae, x_train, y_train)
    #plot_label_clusters(z_mean, y_train)
    
    
    

if __name__ == '__main__':
    main()    
