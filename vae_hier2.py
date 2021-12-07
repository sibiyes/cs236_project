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

from vae3_modules import vae_mnist_simple, fashion_product_simple, fashion_product2, fashion_product3, fashion_product4


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
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
            
            for i, block in enumerate(self.model.middle_blocks_enc):
                block.save(self.save_folder +  "/model_{0}_mb_enc{1}.h5".format(epoch, i+1))
                
            for i, block in enumerate(self.model.middle_blocks_dec):
                block.save(self.save_folder +  "/model_{0}_mb_dec{1}.h5".format(epoch, i+1))
            

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        

class VAE(keras.Model):
    def __init__(self, encoder, decoder, middle_blocks_enc, middle_blocks_dec, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.middle_blocks_enc = middle_blocks_enc
        self.middle_blocks_dec = middle_blocks_dec
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        
        self.kl_loss_tracker = [keras.metrics.Mean(name="kl_loss1")]
        for i, _ in enumerate(self.middle_blocks_enc):
            self.kl_loss_tracker.append(keras.metrics.Mean(name="kl_loss{0}".format(i+2)))

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker] + self.kl_loss_tracker

    def train_step(self, data):
        with tf.GradientTape() as tape:
            kl_loss_all = []
            
            z_mean1, z_log_var1, z = self.encoder(data)
            kl_loss = -0.5 * (1 + z_log_var1 - tf.square(z_mean1) - tf.exp(z_log_var1))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            kl_loss_all.append(kl_loss)
            
            z_out = [z]
            
            for mb in self.middle_blocks_enc:
                z_mean, z_log_var, z = mb(z)
                z_out.append(z)
                
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                
                kl_loss_all.append(kl_loss)
                
            ### reconstruction
            
            mbd = self.middle_blocks_dec[0]
            zs = z_out[0]
            
            _, _, zd = mbd(zs)
            
            for i in range(1, len(z_out)):
                zs = z_out[i]
                mbd = self.middle_blocks_dec[i]
                
                z_concat = tf.concat((zs, zd), axis = 1)
                
                _, _, zd = mbd(z_concat)
            
            reconstruction = self.decoder(zd)
            
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            
            total_loss = reconstruction_loss
            for loss in kl_loss_all:
                total_loss += loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        
        #self.kl_loss_tracker.update_state(kl_loss)
        for tracker, loss in zip(self.kl_loss_tracker, kl_loss_all):
            tracker.update_state(loss)
            
        loss_vals = {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result()
        }
        
        for i, tracker in enumerate(self.kl_loss_tracker):
            loss_vals['kl_loss{0}'.format(i+1)] = tracker.result()
        
        # return {
        #     "loss": self.total_loss_tracker.result(),
        #     "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        #     "kl_loss": self.kl_loss_tracker.result(),
        # }
        
        return loss_vals
        
    # def call(self, x):
    #     z_mean1, z_log_var1, z = self.encoder(x)
    #     
    #     for mb in self.middle_blocks:
    #         z_mean, z_log_var, z = mb(z)
    #         
    #     reconstruction = self.decoder(z)
    #     
    #     return reconstruction
        
        
def latent_middle_block(latent_dim, input_size):
    middle_inputs = keras.Input(shape=(input_size,))
    x = layers.Dense(2*input_size, activation="relu")(middle_inputs)
    x = layers.Dense(2*input_size, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    middle_block = keras.Model(middle_inputs, [z_mean, z_log_var, z], name="middle_block")
    
    return middle_block
        
def main():
    
    ### Encoder ###
    latent_dim = 10
    run = 3

    #data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder/apparel/men'
    data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder_clean/apparel/men'
    
    #data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder/test'
    
    data_set = keras.preprocessing.image_dataset_from_directory(data_folder, label_mode = None, image_size=(128, 128))
    data_set = data_set.map(lambda x: (tf.divide(x, 255)))
    
    encoder, decoder = fashion_product3(latent_dim)
    model_tag = 'model_fashion_hier2'
    
    n_middle = 3
    middle_blocks_enc = []
    
    for _ in range(n_middle):
        mb = latent_middle_block(latent_dim, latent_dim)
        middle_blocks_enc.append(mb)
        
    mb = latent_middle_block(latent_dim, latent_dim)
    middle_blocks_dec = [mb]
    
    for _ in range(n_middle):
        mb = latent_middle_block(latent_dim, 2*latent_dim)
        middle_blocks_dec.append(mb)
        
    
    # middle_inputs = keras.Input(shape=(latent_dim,))
    # x = layers.Dense(2*latent_dim, activation="relu")(middle_inputs)
    # x = layers.Dense(2*latent_dim, activation="relu")(x)
    # z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    # z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    # z = Sampling()([z_mean, z_log_var])
    # middle_block = keras.Model(middle_inputs, [z_mean, z_log_var, z], name="middle_block")
    
    
    model_folder = base_folder + '/output/vae_hier2/models/{0}_z{1}/run{2}'.format(model_tag, latent_dim, run)
    latent_z_folder = base_folder + '/output/vae_hier2/latent_z/{0}_z{1}/run{2}'.format(model_tag, latent_dim, run)
    create_folder(model_folder)
    create_folder(latent_z_folder)
    
    
    epochs = 100
                                
    model_save_callback = CustomSaver(save_folder = model_folder, save_epochs = 25)

    vae = VAE(encoder, decoder, middle_blocks_enc, middle_blocks_dec)
    
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
    # vae.build(input_shape = (None, 128, 128, 3))
    # print(vae.summary())
    # sys.exit(0)
    
    #vae.fit(mnist_digits, epochs = epochs, batch_size = 128, callbacks = [model_save_callback])
    vae.fit(data_set, epochs = epochs, batch_size = 32, callbacks = [model_save_callback])
    
    vae.encoder.save(model_folder +  "/model_final_encoder.h5")
    vae.decoder.save(model_folder + "/model_final_decoder.h5")
    
    for i, block in enumerate(vae.middle_blocks_enc):
        block.save(model_folder +  "/model_final_mb_enc{0}.h5".format(i+1))
        
    for i, block in enumerate(vae.middle_blocks_dec):
        block.save(model_folder +  "/model_final_mb_dec{0}.h5".format(i+1))
    
    sys.exit(0)
    
    

if __name__ == '__main__':
    main()    
