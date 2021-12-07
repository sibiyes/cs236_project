import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

import matplotlib.pyplot as plt
from PIL import Image

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

from pixel_cnn import PixelConvLayer1, PixelConvLayer2, PixelConvLayer3, ResidualBlock

def load_input_image():
    image_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder/apparel/men'
    img_file = image_folder + '/' + '1853.jpg'
    
    image = Image.open(img_file)
    image = image.resize((128, 128))
    image = np.asarray(image)/255.0
    
    # print(image)
    # print(np.min(image), np.max(image))
    # 
    # plt.imshow(image)
    # plt.show()
    
    # image[60:80, 50:70] = np.array([1.0, 1.0, 1.0])
    # 
    # plt.imshow(image)
    # plt.show()
    
    return image

def main():
    model_tag = 'paper_res3_f32'
    run = 1
    model_folder = base_folder + '/output/pixel_cnn/models/{0}/run{1}'.format(model_tag, run)
    
    #pixel_cnn = keras.models.load_model(model_folder + '/model_25.h5', custom_objects = {'PixelConvLayer': PixelConvLayer(filters=32, kernel_size=7, mask_type = 'A'), 'ResidualBlock': ResidualBlock(filters=32)})
    #custom_objects = {'PixelConvLayer': PixelConvLayer(filters=32, kernel_size=7, mask_type = 'A'), 'ResidualBlock': ResidualBlock(filters=32)}
    
    custom_objects = {
        'PixelConvLayer2': PixelConvLayer2(mask_type="A"),
        'ResidualBlock': ResidualBlock(),
        'PixelConvLayer1': PixelConvLayer1(mask_type="B"),
        'PixelConvLayer3': PixelConvLayer3(mask_type = "B")
    }
    
    #pixel_cnn = keras.models.load_model(model_folder + '/model_0.h5')
    pixel_cnn = keras.models.load_model(model_folder + '/model_25.h5', custom_objects = custom_objects)
    
    print(pixel_cnn)
    
    # Create an empty array of pixels.
    batch = 1
    pixels = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
    batch, rows, cols, channels = pixels.shape
    
    print('size', batch, rows, cols, channels)
    
    img_input = load_input_image()
    img_input = img_input.reshape((1,) + (img_input.shape))
    
    print(img_input.shape)
    
    #sys.exit(0)
    
    for row in tqdm(range(60, 80)):
        for col in range(50, 70):
            for channel in range(channels):
                probs = pixel_cnn.predict(img_input)[:, row, col, channel]
                img_input[:, row, col, channel] = tf.math.ceil(probs - tf.random.uniform(probs.shape))
                
    for i, pic in enumerate(img_input):
        pic *= 255.0
        pic = np.clip(pic, 0, 255).astype("uint8")
        
        plt.imshow(pic)
        plt.show()
                
    sys.exit(0)
    

    # # Iterate over the pixels because generation has to be done sequentially pixel by pixel.
    # for row in tqdm(range(rows)):
    #     for col in range(cols):
    #         for channel in range(channels):
    #             # Feed the whole array and retrieving the pixel value probabilities for the next
    #             # pixel.
    #             probs = pixel_cnn.predict(pixels)[:, row, col, channel]
    #             # Use the probabilities to pick pixel values and append the values to the image
    #             # frame.
    #             pixels[:, row, col, channel] = tf.math.ceil(
    #                 probs - tf.random.uniform(probs.shape)
    #            )
                
    # print('PIXELS')
    # print(pixels)
    # print(pixels.shape)

    # def deprocess_image(x):
    #     # Stack the single channeled black and white image to RGB values.
    #     x = np.stack((x, x, x), 2)
    #     # Undo preprocessing
    #     x *= 255.0
    #     # Convert to uint8 and clip to the valid range [0, 255]
    #     x = np.clip(x, 0, 255).astype("uint8")
    #     return x
    # 
    # # Iterate over the generated images and plot them with matplotlib.
    # for i, pic in enumerate(pixels):
    #     keras.preprocessing.image.save_img(
    #         "generated_image_{}.png".format(i), deprocess_image(np.squeeze(pic, -1))
    #    )

    
    for i, pic in enumerate(pixels):
        pic *= 255.0
        pic = np.clip(pic, 0, 255).astype("uint8")
        
        plt.imshow(pic)
        plt.show()
    
if __name__ == '__main__':
    #load_input_image()
    main()
