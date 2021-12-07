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

from bigan_modules import fashion_product2

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
        
class BIGAN(keras.Model):
    def __init__(self, discriminator, encoder, generator, latent_dim):
        super(BIGAN, self).__init__()
        self.discriminator = discriminator
        self.encoder = encoder
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(BIGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.kl_loss_metric = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric, self.kl_loss_metric]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)
        
        # Encode real images to latent variables
        
        #real_latent_vectors = self.encoder(real_images)
        
        #########################
        
        latent_mean, latent_log_var, real_latent_vectors = self.encoder(real_images)
        
        kl_loss = -0.5 * (1 + latent_log_var - tf.square(latent_mean) - tf.exp(latent_log_var))
        #print('kl loss', kl_loss)
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        
        ##########################

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)
        combined_latent_var = tf.concat([random_latent_vectors, real_latent_vectors], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator([combined_images, combined_latent_var])
            d_loss = self.loss_fn(labels, predictions)
            
            total_loss = d_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.discriminator.trainable_weights + self.encoder.trainable_weights,
                                        unconnected_gradients=tf.UnconnectedGradients.ZERO)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights + self.encoder.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator([self.generator(random_latent_vectors), random_latent_vectors])
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        self.kl_loss_metric.update_state(kl_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "kl_loss": self.kl_loss_metric.result()
        }


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img, save_epochs, save_folder, latent_dim=128):
        self.num_img = num_img
        self.save_epochs = save_epochs
        self.save_folder = save_folder
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.save_epochs) == 0:
            random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
            generated_images = self.model.generator(random_latent_vectors)
            generated_images *= 255
            generated_images.numpy()
            for i in range(self.num_img):
                img = keras.preprocessing.image.array_to_img(generated_images[i])
                img.save(self.save_folder + "/generated_img_%03d_%d.png" % (epoch, i))
            
            
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
            self.model.generator.save(self.save_folder + "/model_{}_generator.h5".format(epoch))


def main():
    
    ### Encoder ###
    latent_dim = 10
    run = 1

    #data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder/apparel/men'
    data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder_clean/apparel/men'
    #data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder/test'
    data_set = keras.preprocessing.image_dataset_from_directory(data_folder, label_mode = None, image_size=(128, 128))
    data_set = data_set.map(lambda x: (tf.divide(x, 255)))
    
    encoder, decoder, discriminator = fashion_product2(latent_dim)
    model_tag = 'bigan2'
    
    model_folder = base_folder + '/output/bigan2/models/{0}_z{1}/run{2}'.format(model_tag, latent_dim, run)
    image_output_folder = base_folder + '/output/bigan2/image_gen/{0}_z{1}/run{2}'.format(model_tag, latent_dim, run)
    create_folder(model_folder)
    create_folder(image_output_folder)
    
    
    epochs = 100  # In practice, use ~100 epochs

    bigan = BIGAN(discriminator = discriminator, encoder = encoder, generator = decoder, latent_dim=latent_dim)
    bigan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )
    
    model_save_callback = CustomSaver(save_folder = model_folder, save_epochs = 25)
    img_save_callback = GANMonitor(num_img=5, save_epochs = 10, save_folder = image_output_folder, latent_dim=latent_dim)

    bigan.fit(
        data_set, epochs=epochs, callbacks=[model_save_callback, img_save_callback]
    )
    
    #bigan.fit(data_set, epochs=epochs)
    
    

if __name__ == '__main__':
    main()    
