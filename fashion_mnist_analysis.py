import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from keras.datasets.fashion_mnist import load_data

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)


def main():
    (train_X, train_Y), (test_X, test_Y) = load_data()
    
    # summarize the shape of the dataset
    print('Train', train_X.shape, train_Y.shape)
    print('Test', test_X.shape, test_Y.shape)
    
    # plt.imshow(train_X[0], cmap='gray')
    # plt.show()
    
    print(train_Y)
    print(type(train_Y))
    
    classes = np.arange(10)
    print(classes)
    
    for c in classes:
        print(c, np.sum(train_Y == c))
    
    print(len(train_Y))
    
    data_train_folder = base_folder + '/data/fashion_mnist/images/train'
    for c in classes:
        data_train_class_folder = data_train_folder + '/' + str(c)
        if (not os.path.exists(data_train_class_folder)):
            os.makedirs(data_train_class_folder)
    
    for i in range(len(train_Y)):
        print(i)
        img = train_X[i]
        # print(img)
        # print(np.shape(img))
        # print(train_Y[i])
        
        img_save_folder = data_train_folder + '/' + str(train_Y[i])
        #print(img_save_folder)
        plt.imshow(img, cmap='gray')
        plt.savefig(img_save_folder + '/img{0}.jpg'.format(i))
        
        #sys.exit(0)
    
    
if __name__ == '__main__':
    main()
