import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
from PIL import Image

from sklearn.decomposition import PCA

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        
def load_image(img_path = None):
    if (img_path is None):
        data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder/apparel/men'
        img_file = os.listdir(data_folder)[200]
        img_path = data_folder + '/' + img_file
    
    img = Image.open(img_path)
    img = img.resize((128, 128))
    img = np.asarray(img)/255.0
    
    #print(np.shape(img))
    
    # plt.imshow(img)
    # plt.show()
    
    return img
    
    
def gen_new_sample_mnist():
    model_folder = base_folder + '/output/vae3/models/model_simple_z{0}/run{1}'.format(5, 1)
    encoder = keras.models.load_model(model_folder + '/model_10_encoder.h5', custom_objects = {'Sampling': Sampling})
    decoder = keras.models.load_model(model_folder + '/model_10_decoder.h5')
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    
    _, _, z = encoder.predict(x_train)
    
    digit_size = 28
    z_sample = np.array([z[50, :]])
    x_decoded = decoder.predict(z_sample)
    digit = x_decoded[0].reshape(digit_size, digit_size)
    
    plt.imshow(digit, cmap="Greys_r")
    plt.show()
    
def gen_new_sample_vae_hier1(image_inputs, digit_size, n_channel, model_tag, latent_dim, run, m, title):
    model_folder = base_folder + '/output/vae_hier1/models/{0}_z{1}/run{2}'.format(model_tag, latent_dim, run)
    
    encoder = keras.models.load_model(model_folder + '/model_final_encoder.h5', custom_objects = {'Sampling': Sampling})
    middle_blocks = []
    
    for i in range(m):
        mb = keras.models.load_model(model_folder + '/model_final_mb{0}.h5'.format(i+1), custom_objects = {'Sampling': Sampling})
        middle_blocks.append(mb)
        
    print(middle_blocks)
    
    decoder = keras.models.load_model(model_folder + '/model_final_decoder.h5', custom_objects = {'Sampling': Sampling})
    
    n_row = len(image_inputs)
    n_col = 5
    
    
    for j in range(n_row):
        
        img = image_inputs[j]
        img_input = img.reshape((1,) + np.shape(img))
        
        ax = plt.subplot(n_row, n_col, j*n_col + 1)
        ax.axis('off')
        ax.title.set_text('Input Image')
        plt.imshow(img)
        
        _, _, z_sample = encoder.predict(img_input)
        
        for mb in middle_blocks:
            z_mean, _, z_sample = mb(z_sample)
        
        x_decoded = decoder.predict(z_sample)
        sample = x_decoded[0].reshape(digit_size, digit_size, n_channel)
        
        ax = plt.subplot(n_row, n_col, j*n_col + 2)
        ax.axis('off')
        ax.title.set_text('Decode - No added noise')
        plt.imshow(sample)
        
        for i in range(n_col-2):
            _, _, z_sample = encoder.predict(img_input)
            
            for mb in middle_blocks:
                z_mean, _, z_sample = mb(z_sample)
                
            epsilon = tf.keras.backend.random_normal(shape=(1, latent_dim))
            
            z_sample += 0.2*epsilon
            
            x_decoded = decoder.predict(z_sample)
            sample = x_decoded[0].reshape(digit_size, digit_size, n_channel)
            
            ax = plt.subplot(n_row, n_col, j*n_col + i+3)
            ax.axis('off')
            ax.title.set_text('Decode - added noise')
            plt.imshow(sample)
        
    plt.suptitle(title)
    plt.show()
    plt.tight_layout()
    

    
    
def gen_new_sample(image_inputs, digit_size, n_channel, model_tag, latent_dim, run, title):
    model_folder = base_folder + '/output/vae3/models/{0}_z{1}/run{2}'.format(model_tag, latent_dim, run)
    encoder = keras.models.load_model(model_folder + '/model_final_encoder.h5', custom_objects = {'Sampling': Sampling})
    decoder = keras.models.load_model(model_folder + '/model_final_decoder.h5')
    
    n_row = len(image_inputs)
    n_col = 5
    
    for j in range(n_row):
        # img_file = os.listdir(data_folder)[img_id[j]]
        # img_path = data_folder + '/' + img_file
        # 
        # img = load_image(img_path)
        # img_input = img.reshape((1,) + np.shape(img))
        
        img = image_inputs[j]
        img_input = img.reshape((1,) + np.shape(img))
        
        ax = plt.subplot(n_row, n_col, j*n_col + 1)
        ax.axis('off')
        ax.title.set_text('Input Image')
        plt.imshow(img)
        
        _, _, z_sample = encoder.predict(img_input)
        x_decoded = decoder.predict(z_sample)
        sample = x_decoded[0].reshape(digit_size, digit_size, n_channel)
        
        ax = plt.subplot(n_row, n_col, j*n_col + 2)
        ax.axis('off')
        ax.title.set_text('Decode - No added noise')
        plt.imshow(sample)
        
        for i in range(n_col-2):
            _, _, z_sample = encoder.predict(img_input)
            epsilon = tf.keras.backend.random_normal(shape=(1, latent_dim))
            
            z_sample += 0.2*epsilon
            
            x_decoded = decoder.predict(z_sample)
            sample = x_decoded[0].reshape(digit_size, digit_size, n_channel)
            
            ax = plt.subplot(n_row, n_col, j*n_col + i+3)
            ax.axis('off')
            ax.title.set_text('Decode - added noise')
            plt.imshow(sample)
        
    plt.suptitle(title)
    plt.show()
    plt.tight_layout()
    
def gen_new_sample_fashion(dataset):
    if (dataset == 'fashion_product'):
        model_tag = 'model_fashion_simple'
        latent_dim = 10
        run = 3
        
        digit_size = 128
        n_channel = 3
        title = 'Fashion Product  - latent_dim: {0}'.format(latent_dim)
        
        img_id_vals = [75, 200, 430, 976]
        
        #data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder/apparel/men'
        data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder_clean/apparel/men'
        
        image_inputs = []
        for img_id in img_id_vals:
            img_file = os.listdir(data_folder)[img_id]
            img_path = data_folder + '/' + img_file
            
            img = load_image(img_path)
            
            image_inputs.append(img)
    
    if (dataset == 'fashion_mnist'):
        latent_dim = 10
        model_folder = base_folder + '/output/vae3/models/model_simple_z{0}/run{1}'.format(latent_dim, 1)
        encoder = keras.models.load_model(model_folder + '/model_final_encoder.h5', custom_objects = {'Sampling': Sampling})
        decoder = keras.models.load_model(model_folder + '/model_final_decoder.h5')
        
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = np.expand_dims(x_train, -1).astype("float32") / 255
        
        digit_size = 28
        n_channel = 1
        title = 'Fashion MNIST  - latent_dim: {0}'.format(latent_dim)
        
        img_id_vals = [150, 58006, 5423, 467]
        
        image_inputs = []
        
        for img_id in img_id_vals:
            img = x_train[img_id]
            image_inputs.append(img)
            
            #sys.exit(0)
        
    
    gen_new_sample(image_inputs, digit_size, n_channel, model_tag, latent_dim, run, title)
    #gen_new_sample_vae_hier1(image_inputs, digit_size, n_channel, model_tag, latent_dim, run, 2, title)
    
    
def main():
    
    gen_new_sample_fashion('fashion_product')
    
if __name__ == '__main__':
    main()
