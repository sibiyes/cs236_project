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
from sklearn.manifold import TSNE

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
        
def fit_pca(data):
    pca = PCA(n_components=2)
    pca_transform = pca.fit_transform(data)
    
    return pca_transform
    
def fit_tsne(data):
    tsne = TSNE(n_components=2, learning_rate='auto', init='random')
    tsne_transform = tsne.fit_transform(data)
    
    return tsne_transform
        
def plot_label_clusters(z_mean, labels):
    # display a 2D plot of the digit classes in the latent space
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
    
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
    
def gen_new_sample(image_inputs, digit_size, n_channel, latent_dim, encoder, decoder, title):
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
        latent_dim = 10
        model_folder = base_folder + '/output/vae3/models/model_fashion_simple_z{0}/run{1}'.format(latent_dim, 1)
        encoder = keras.models.load_model(model_folder + '/model_final_encoder.h5', custom_objects = {'Sampling': Sampling})
        decoder = keras.models.load_model(model_folder + '/model_final_decoder.h5')
        
        digit_size = 128
        n_channel = 3
        title = 'Fashion Product  - latent_dim: {0}'.format(latent_dim)
        
        img_id_vals = [75, 200, 430, 976]
        
        data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder/apparel/men'
        
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
        
    
    gen_new_sample(image_inputs, digit_size, n_channel, latent_dim, encoder, decoder, title)
    
    #sys.exit(0)
        
    
    
def gen_z_fashion():
    latent_dim = 10
    model_folder = base_folder + '/output/vae3/models/model_fashion_simple_z{0}/run{1}'.format(latent_dim, 1)
    encoder = keras.models.load_model(model_folder + '/model_final_encoder.h5', custom_objects = {'Sampling': Sampling})
    
    data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder/apparel/men'
    
    z_mean_all = []
    z_sample_all = []
    
    for img_file in os.listdir(data_folder):
        print(img_file)
        img_path = data_folder + '/' + img_file
        
        img = load_image(img_path)
        #print(img)
        
        img_input = img.reshape((1,) + np.shape(img))
        
        z_mean, _, z_sample = encoder.predict(img_input)
        
        print(z_mean)
        print(z_sample)
        
        z_mean_all.append([img_file] + list(z_mean[0]))
        z_sample_all.append([img_file] + list(z_sample[0]))
        
        #sys.exit(0)
        
    z_mean_all = pd.DataFrame(z_mean_all, columns = ['img_f'] + ['z{0}'.format(i) for i in range(latent_dim)])
    z_sample_all = pd.DataFrame(z_sample_all, columns = ['img_f'] + ['z{0}'.format(i) for i in range(latent_dim)])
    
    output_folder = base_folder + '/output/vae3/latent_z/model_fashion_simple_z{0}/run{1}'.format(latent_dim, 1)
    
    z_mean_all.to_csv(output_folder + '/' + 'z_mean_final.csv', index = None)
    z_sample_all.to_csv(output_folder + '/' + 'z_sample_final.csv', index = None)
    
    print(z_mean_all)
    print(z_sample_all)
    
def gen_z_mnist():
    latent_dim = 10
    model_folder = base_folder + '/output/vae3/models/model_simple_z{0}/run{1}'.format(latent_dim, 1)
    encoder = keras.models.load_model(model_folder + '/model_final_encoder.h5', custom_objects = {'Sampling': Sampling})
    
    output_folder = base_folder + '/output/vae3/latent_z/model_simple_z{0}/run{1}'.format(latent_dim, 1)
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255
    
    z_mean_all = []
    z_sample_all = []
    
    for i in range(x_train.shape[0]):
        img = x_train[i]
        img_input = img.reshape((1,) + np.shape(img))
        
        z_mean, _, z_sample = encoder.predict(img_input)
        print(z_mean)
        
        z_mean_all.append(list(z_mean[0]))
        z_sample_all.append(list(z_sample[0]))
    
    z_mean_all = pd.DataFrame(z_mean_all, columns = ['z{0}'.format(i) for i in range(latent_dim)])
    z_sample_all = pd.DataFrame(z_sample_all, columns = ['z{0}'.format(i) for i in range(latent_dim)])
    
    z_mean_all.to_csv(output_folder + '/' + 'z_mean_final.csv', index = None)
    z_sample_all.to_csv(output_folder + '/' + 'z_sample_final.csv', index = None)
    
    z_mean_all = []
    z_sample_all = []
    
    for i in range(x_test.shape[0]):
        img = x_test[i]
        img_input = img.reshape((1,) + np.shape(img))
        
        z_mean, _, z_sample = encoder.predict(img_input)
        print(z_mean)
        
        z_mean_all.append(list(z_mean[0]))
        z_sample_all.append(list(z_sample[0]))
    
    z_mean_all = pd.DataFrame(z_mean_all, columns = ['z{0}'.format(i) for i in range(latent_dim)])
    z_sample_all = pd.DataFrame(z_sample_all, columns = ['z{0}'.format(i) for i in range(latent_dim)])
    
    z_mean_all.to_csv(output_folder + '/' + 'z_mean_test_final.csv', index = None)
    z_sample_all.to_csv(output_folder + '/' + 'z_sample_test_final.csv', index = None)
    
    
    
def plot_mnist():
    model_folder = base_folder + '/output/vae3/models/model_simple_z{0}/run{1}'.format(5, 1)
    encoder = keras.models.load_model(model_folder + '/model_10_encoder.h5', custom_objects = {'Sampling': Sampling})
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    
    z_mean, _, z_sample = encoder.predict(x_train)
    #z_mean = pd.DataFrame(z_mean, columns = ['z{0}'.format(i+1) for i in range(latent_dim)])
    #np.savetxt(base_folder + '/output/vae3/latent_z', z_mean, delimiter = ",")
    print(z_sample)
    
    z_pca = fit_pca(z_sample)
    plot_label_clusters(z_pca, y_train)
    
    # z_tsne = fit_tsne(z_sample)
    # plot_label_clusters(z_tsne, y_train)
    
def plot_fashion_product():
    # model_folder = base_folder + '/output/vae3/models/model_fashion_simple_z{0}/run{1}'.format(10, 1)
    # encoder = keras.models.load_model(model_folder + '/model_final_encoder.h5', custom_objects = {'Sampling': Sampling})
    # 
    # data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder/apparel/men'
    # data_set = keras.preprocessing.image_dataset_from_directory(data_folder, label_mode = None, image_size=(128, 128))
    # data_set = data_set.map(lambda x: (tf.divide(x, 255)))
    # 
    # z_mean, _, _ = encoder.predict(data_set)
    
    latent_dim = 20
    run = 1
    #latent_z_file = base_folder + '/output/vae3/latent_z/model_fashion_simple_z{0}/run{1}/z_sample_final.csv'.format(latent_dim, run)
    latent_z_file = base_folder + '/output/vae3/latent_z/model_fashion3_z{0}/run{1}/z_sample_final.csv'.format(latent_dim, run)
    
    latent_z = pd.read_csv(latent_z_file)
    latent_z = latent_z.iloc[:, 1:]
    print(latent_z)
    
    y_train = np.ones(latent_z.shape[0])
    
    z_pca = fit_pca(latent_z)
    plot_label_clusters(z_pca, y_train)
    
    z_tsne = fit_tsne(latent_z)
    plot_label_clusters(z_tsne, y_train)

def main():
    print(5)
    #plot_mnist()
    plot_fashion_product()
    #gen_new_sample_mnist()
    #gen_new_sample_fashion('fashion_mnist')
    
    #gen_z_fashion()
    #gen_z_mnist()
    
    #load_image()
    
    
    
    
if __name__ == '__main__':
    main()
