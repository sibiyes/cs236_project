import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

def main():
    input_images = [
        base_folder + '/plots/fashion_product/output_z10.png',
        base_folder + '/plots/fashion_product/output_z20.png'
    ]
    
    images_all = []
    for img_path in input_images:
        img = Image.open(img_path)
        img = np.asarray(img)
        images_all.append(img)
        
        print(img.shape)
        
    images_concat = np.concatenate(images_all, axis = 1)
    
    print(images_concat.shape)
    
    plt.imshow(images_concat)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
