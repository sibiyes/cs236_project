import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from collections import defaultdict

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
    
    
        
def gen_z_fashion(model_tag = None, latent_dim = None, run = None):
    # latent_dim = 10
    # model_tag = 'model_fashion_simple'
    # run = 1
    
    #model_folder = base_folder + '/output/vae3/models/model_fashion_simple_z{0}/run{1}'.format(latent_dim, 1)
    model_folder = base_folder + '/output/vae3/models/{0}_z{1}/run{2}'.format(model_tag, latent_dim, run)
    
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
    
    output_folder = base_folder + '/output/vae3/latent_z/{0}_z{1}/run{2}'.format(model_tag, latent_dim, run)
    #output_folder = base_folder + '/output/vae3/latent_z/model_fashion_simple_z{0}/run{1}'.format(latent_dim, 1)
    
    z_mean_all.to_csv(output_folder + '/' + 'z_mean_final.csv', index = None)
    z_sample_all.to_csv(output_folder + '/' + 'z_sample_final.csv', index = None)
    
    print(z_mean_all)
    print(z_sample_all)
    

def gen_z_fashion_hier1(model_tag = None, m = None, latent_dim = None, run = None, data = None):
    # latent_dim = 20
    # model_tag = 'model_fashion_hier1'
    # run = 1
    # m = 2
    
    #model_folder = base_folder + '/output/vae3/models/model_fashion_simple_z{0}/run{1}'.format(latent_dim, 1)
    model_folder = base_folder + '/output/vae_hier1/models/{0}_z{1}/run{2}'.format(model_tag, latent_dim, run)
    
    encoder = keras.models.load_model(model_folder + '/model_final_encoder.h5', custom_objects = {'Sampling': Sampling})
    middle_blocks = []
    
    for i in range(m):
        mb = keras.models.load_model(model_folder + '/model_final_mb{0}.h5'.format(i+1), custom_objects = {'Sampling': Sampling})
        middle_blocks.append(mb)
        
    print(middle_blocks)
    
    if (data == 'men'):
        data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder/apparel/men'
    if (data == 'men_clean'):
        data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder_clean/apparel/men'
    
    z_mean_all = []
    z_sample_all = []
    
    mb_mean_all = []
    mb_sample_all = []
    
    for _ in range(m):
        mb_mean_all.append([])
        mb_sample_all.append([])
    
    c = 0
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
        
        for i, mb in enumerate(middle_blocks):
            z_mean, _, z_sample = mb(z_sample)
            
            mb_mean_all[i].append([img_file] + list(z_mean[0].numpy()))
            mb_sample_all[i].append([img_file] + list(z_sample[0].numpy()))
        
        c += 1
        
        # if (c > 100):
        #     break
        
        #break
        
    z_mean_all = pd.DataFrame(z_mean_all, columns = ['img_f'] + ['z{0}'.format(i) for i in range(latent_dim)])
    z_sample_all = pd.DataFrame(z_sample_all, columns = ['img_f'] + ['z{0}'.format(i) for i in range(latent_dim)])
    
    output_folder = base_folder + '/output/vae_hier1/latent_z/{0}_z{1}/run{2}'.format(model_tag, latent_dim, run)
    #output_folder = base_folder + '/output/vae_hier1/latent_z/model_fashion_simple_z{0}/run{1}'.format(latent_dim, 1)
    
    z_mean_all.to_csv(output_folder + '/' + 'z_mean_final.csv', index = None)
    z_sample_all.to_csv(output_folder + '/' + 'z_sample_final.csv', index = None)
    
    print(z_mean_all)
    print(z_sample_all)
    
    print('---------------------')
    
    for i, mb_mean in enumerate(mb_mean_all):
        mb_mean = pd.DataFrame(mb_mean, columns = ['img_f'] + ['z{0}'.format(i) for i in range(latent_dim)])
        print(mb_mean)
        mb_mean.to_csv(output_folder + '/' + 'mb{0}_mean_final.csv'.format(i+1), index = None)
        
    for i, mb_sample in enumerate(mb_sample_all):
        mb_sample = pd.DataFrame(mb_mean, columns = ['img_f'] + ['z{0}'.format(i) for i in range(latent_dim)])
        print(mb_mean)
        mb_sample.to_csv(output_folder + '/' + 'mb{0}_sample_final.csv'.format(i+1), index = None)
        
        
def gen_z_fashion_hier2(model_tag = None, m = None, latent_dim = None, run = None, data = None):
    latent_dim = 10
    model_tag = 'model_fashion_hier2'
    run = 1
    m = 2
    
    #model_folder = base_folder + '/output/vae3/models/model_fashion_simple_z{0}/run{1}'.format(latent_dim, 1)
    model_folder = base_folder + '/output/vae_hier2/models/{0}_z{1}/run{2}'.format(model_tag, latent_dim, run)
    
    encoder = keras.models.load_model(model_folder + '/model_final_encoder.h5', custom_objects = {'Sampling': Sampling})
    middle_blocks = []
    
    for i in range(m):
        mb = keras.models.load_model(model_folder + '/model_final_mb_enc{0}.h5'.format(i+1), custom_objects = {'Sampling': Sampling})
        middle_blocks.append(mb)
        
    print(middle_blocks)
    
    data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder/apparel/men'
    
    z_mean_all = []
    z_sample_all = []
    
    mb_mean_all = []
    mb_sample_all = []
    
    for _ in range(m):
        mb_mean_all.append([])
        mb_sample_all.append([])
    
    c = 0
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
        
        for i, mb in enumerate(middle_blocks):
            z_mean, _, z_sample = mb(z_sample)
            
            mb_mean_all[i].append([img_file] + list(z_mean[0].numpy()))
            mb_sample_all[i].append([img_file] + list(z_sample[0].numpy()))
        
        c += 1
        
        # if (c > 100):
        #     break
        
        #break
        
    z_mean_all = pd.DataFrame(z_mean_all, columns = ['img_f'] + ['z{0}'.format(i) for i in range(latent_dim)])
    z_sample_all = pd.DataFrame(z_sample_all, columns = ['img_f'] + ['z{0}'.format(i) for i in range(latent_dim)])
    
    output_folder = base_folder + '/output/vae_hier2/latent_z/{0}_z{1}/run{2}'.format(model_tag, latent_dim, run)
    #output_folder = base_folder + '/output/vae_hier1/latent_z/model_fashion_simple_z{0}/run{1}'.format(latent_dim, 1)
    
    z_mean_all.to_csv(output_folder + '/' + 'z_mean_final.csv', index = None)
    z_sample_all.to_csv(output_folder + '/' + 'z_sample_final.csv', index = None)
    
    print(z_mean_all)
    print(z_sample_all)
    
    print('---------------------')
    
    for i, mb_mean in enumerate(mb_mean_all):
        mb_mean = pd.DataFrame(mb_mean, columns = ['img_f'] + ['z{0}'.format(i) for i in range(latent_dim)])
        print(mb_mean)
        mb_mean.to_csv(output_folder + '/' + 'mb_enc{0}_mean_final.csv'.format(i+1), index = None)
        
    for i, mb_sample in enumerate(mb_sample_all):
        mb_sample = pd.DataFrame(mb_mean, columns = ['img_f'] + ['z{0}'.format(i) for i in range(latent_dim)])
        print(mb_mean)
        mb_sample.to_csv(output_folder + '/' + 'mb_enc{0}_sample_final.csv'.format(i+1), index = None)

    #sys.exit(0)

def main():
    print(5)
    
    # model_info = [
    #     ('model_fashion3', 20, 1),
    #     ('model_fashion4', 10, 1),
    #     ('model_fashion5', 10, 1)
    # ]
    
    # for model_tag, latent_dim, run in model_info:
    #     gen_z_fashion(model_tag, latent_dim, run)
    
    # model_info = [
    #     ('model_fashion_hier1', 2, 10, 1, 'men_clean'),
    #     ('model_fashion_hier1', 2, 20, 1, 'men_clean')
    # ]
    # 
    # 
    # for model_tag, m, latent_dim, run, data in model_info:
    #     gen_z_fashion_hier1(model_tag ,m, latent_dim, run, data)
        
    model_info = [
        ('model_fashion_hier2', 2, 10, 1, 'men_clean'),
        ('model_fashion_hier2', 2, 10, 2, 'men_clean'),
        ('model_fashion_hier2', 3, 10, 3, 'men_clean')    
    ]
    
    #('model_fashion_hier2', 2, 20, 1, 'men_clean')
    
    for model_tag, m, latent_dim, run, data in model_info:
        gen_z_fashion_hier2(model_tag ,m, latent_dim, run, data)
        
    
    
    #gen_z_mnist()
    
if __name__ == '__main__':
    main()
    
