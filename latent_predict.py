import os
import sys

import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

def run_model(train_X, train_Y, test_X, test_Y):
    lr_model = LogisticRegression(random_state=0)
    lr_model.fit(train_X, train_Y)

def predict_latent_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    latent_dim = 5
    latent_z_train_file = base_folder + '/output/vae3/latent_z/model_simple_z{0}/run{1}/z_sample_final.csv'.format(latent_dim, 1)
    latent_z_test_file = base_folder + '/output/vae3/latent_z/model_simple_z{0}/run{1}/z_sample_test_final.csv'.format(latent_dim, 1)
    
    latent_z_train = pd.read_csv(latent_z_train_file)
    latent_z_test = pd.read_csv(latent_z_test_file)
    
    print(latent_z_train)
    print(latent_z_test)
    
    
    lr_model = LogisticRegression(random_state = 0, max_iter = 500).fit(latent_z_train, y_train)
    y_predict_train = lr_model.predict(latent_z_train)
    y_predict_test = lr_model.predict(latent_z_test)
    
    print(y_predict_train)
    
    conf_matrix_train = confusion_matrix(y_train, y_predict_train)
    accuracy_train = accuracy_score(y_train, y_predict_train)
    print(conf_matrix_train)
    print(accuracy_train)
    
    conf_matrix_test = confusion_matrix(y_test, y_predict_test)
    accuracy_test = accuracy_score(y_test, y_predict_test)
    print(conf_matrix_test)
    print(accuracy_test)
    
def data_balance(data, target_attrib):
    print(data)
    
    label_count = data[[target_attrib]].groupby(target_attrib).agg({target_attrib: 'count'})
    print(label_count)
    max_count = label_count.max().values[0]
    
    print(max_count)
    
    data_resample_all = []
    
    for group, data_group in data.groupby(target_attrib):
        #print(group)
        #print(data_agg)
        
        rows, _ = data_group.shape
        
        n = int(np.ceil(max_count/rows))
        
        print(n)
        
        data_resample = [data_group]*n
        data_resample = pd.concat(data_resample)
        data_resample = data_resample.iloc[:max_count, :]

        print(data_resample)
        
        data_resample_all.append(data_resample)

    data_resample_all = pd.concat(data_resample_all)
    
    print(data_resample_all)
    print(data_resample_all[[target_attrib]].groupby(target_attrib).agg({target_attrib: 'count'}))
    
    return data_resample_all
    
def predict_latent_fashion_product():
    latent_dim = 20
    run = 1
    #latent_z_file = base_folder + '/output/vae3/latent_z/model_fashion_simple_z{0}/run{1}/z_sample_final.csv'.format(latent_dim, run)
    latent_z_file = base_folder + '/output/vae3/latent_z/model_fashion3_z{0}/run{1}/z_sample_final.csv'.format(latent_dim, run)
    
    latent_z = pd.read_csv(latent_z_file)
    
    latent_z['img_f'] = latent_z['img_f'].apply(lambda x: x.split('.')[0])
    print(latent_z)
    
    style_info_all = pd.read_csv(base_folder + '/data/fashion_product/fashion-dataset/styles_all.csv')
    style_info_all['img_f'] = style_info_all['img_f'].apply(lambda x: x.split('.')[0])
    
    style_subset = style_info_all[style_info_all['master_category'] == 'Apparel'][style_info_all['sub_category'] == 'Topwear'][style_info_all['gender'] == 'Men']
    print(style_subset)
    
    #attribute = 'base_color'
    attribute = 'fit'
    #attribute = 'article_type_name'
    print(style_subset[[attribute]].groupby(attribute).agg({attribute: 'count'}))
    
    style_subset = style_subset[style_subset['fit'] != 'Loose']
    style_subset = style_subset[style_subset['fit'] != 'Tailored Fit']
    
    print(style_subset[[attribute]].groupby(attribute).agg({attribute: 'count'}))
    
    
    
    latent_z_style = pd.merge(latent_z, style_subset[['img_f', attribute]])
    latent_z_style = latent_z_style.dropna().reset_index(drop = True)
    print(latent_z_style)
    
    n, _ = latent_z_style.shape
    
    print(n)
    
    train_size = int(0.8*n)
    ind = list(latent_z_style.index)
    np.random.shuffle(ind)

    train_ind = ind[:train_size]
    test_ind = ind[train_size:]
        
    # print(latent_z_style.iloc[np.array([1, 2, 3]), :])
    # print(latent_z_style.iloc[train_ind, :])
    # sys.exit(0)
    
    data_train = latent_z_style.iloc[train_ind, :]
    data_test = latent_z_style.iloc[test_ind, :]
    
    # print(data_train)
    # print(data_test)
    
    data_train = data_balance(data_train, attribute)
    
    print(data_train)
    
    feature_cols = ['z{}'.format(i) for i in range(latent_dim)]
    print(feature_cols)
    
    
    lr_model = LogisticRegression(random_state = 0, max_iter = 500).fit(data_train[feature_cols], data_train[attribute])
    y_predict_train = lr_model.predict(data_train[feature_cols])
    y_predict_test = lr_model.predict(data_test[feature_cols])
    
    # y_predict_prob_train = lr_model.predict_proba(data_train[feature_cols])
    # y_predict_prob_test = lr_model.predict_proba(data_test[feature_cols])
    # 
    # print(y_predict_prob_train)
    # sys.exit(0)
    
    print(y_predict_train)
    
    conf_matrix_train = confusion_matrix(data_train[attribute], y_predict_train)
    accuracy_train = accuracy_score(data_train[attribute], y_predict_train)
    print(conf_matrix_train)
    print(accuracy_train)
    
    conf_matrix_test = confusion_matrix(data_test[attribute], y_predict_test)
    accuracy_test = accuracy_score(data_test[attribute], y_predict_test)
    print(conf_matrix_test)
    print(accuracy_test)
    
    
def main():
    #predict_latent_mnist()
    predict_latent_fashion_product()
    
    
    
if __name__ == '__main__':
    main()
