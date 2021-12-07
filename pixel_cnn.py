import sys
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

### https://keras.io/examples/generative/pixelcnn/#build-the-model-based-on-the-original-paper

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

class CustomSaver(keras.callbacks.Callback):
    def __init__(self, save_epochs, save_folder):
        super(CustomSaver, self).__init__()
        self.save_epochs = save_epochs
        self.save_folder = save_folder
        
    def on_epoch_end(self, epoch, logs={}):
        
        if (epoch % self.save_epochs) == 0:  # or save after some epoch, each k-th epoch etc.
            print('saving checkpoint (epoch {0}) ...'.format(epoch))
            self.model.save(self.save_folder +  "/model_{}.h5".format(epoch))


# The first layer is the PixelCNN layer. This layer simply
# builds on the 2D convolutional layer, but includes masking.
class PixelConvLayer1(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer1, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(filters = 16, kernel_size=3, activation="relu", padding="same")
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "mask_type": self.mask_type,
            "conv": self.conv,
        })
        
        return config

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)



class PixelConvLayer2(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer2, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(filters = 32, kernel_size=7, activation="relu", padding="same")
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "mask_type": self.mask_type,
            "conv": self.conv,
        })
        
        return config

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)
        
        
class PixelConvLayer3(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer3, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(filters = 32, kernel_size=1, strides=1, activation="relu", padding="valid")
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "mask_type": self.mask_type,
            "conv": self.conv,
        })
        
        return config

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)

# Next, we build our residual block layer.
# This is just a normal residual block, but based on the PixelConvLayer.
class ResidualBlock(keras.layers.Layer):
    # def __init__(self, **kwargs):
    #     super(ResidualBlock, self).__init__(**kwargs)
    #     self.conv1 = keras.layers.Conv2D(filters = 32, kernel_size=1, activation="relu")
    #     self.pixel_conv = PixelConvLayer1(mask_type="B")
    #     self.conv2 = keras.layers.Conv2D(filters = 32, kernel_size=1, activation="relu")
    #     
    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         "conv1": self.conv1,
    #         "pixel_conv": self.pixel_conv,
    #         "conv2": self.conv2
    #     })
    #     
    #     return config
        
    def __init__(self, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.conv1 = keras.layers.Conv2D(filters = 32, kernel_size=1, activation="relu")
        self.pixel_conv = PixelConvLayer1(mask_type="B")
        self.conv2 = keras.layers.Conv2D(filters = 32, kernel_size=1, activation="relu")
        
    def get_config(self):
        config = super().get_config()
        # config.update({
        #     "conv1": self.conv1,
        #     "pixel_conv": self.pixel_conv,
        #     "conv2": self.conv2
        # })
        
        return config

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])


def main():

    # # Model / data parameters
    # #num_classes = 10
    # input_shape = (28, 28, 1)
    # 
    # # The data, split between train and test sets
    # (x, _), (y, _) = keras.datasets.fashion_mnist.load_data()
    # 
    # # Concatenate all of the images together
    # data = np.concatenate((x, y), axis=0)
    # # Round all pixel values less than 33% of the max 256 value to 0
    # # anything above this value gets rounded up to 1 so that all values are either
    # # 0 or 1
    # data = np.where(data < (0.33 * 256), 0, 1)
    # data = data.astype(np.float32)
    
    ###################
    
    data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder/apparel/men'
    #data_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder/test'
    data = keras.preprocessing.image_dataset_from_directory(data_folder, label_mode = None, image_size=(128, 128))
    data = data.map(lambda x: (tf.divide(x, 255)))
    
    data = tf.data.Dataset.zip((data, data))

    input_shape = (128, 128, 3)
    
    
    #####################
    
    model_tag = 'paper_res3_f32'
    run = 1
    model_folder = base_folder + '/output/pixel_cnn/models/{0}/run{1}'.format(model_tag, run)
    create_folder(model_folder)
    
    model_save_callback = CustomSaver(save_folder = model_folder, save_epochs = 25)
    
    epochs = 100
    
    n_residual_blocks = 3

    inputs = keras.Input(shape=input_shape)
    x = PixelConvLayer2(mask_type = "A")(inputs)

    for _ in range(n_residual_blocks):
        x = ResidualBlock()(x)

    for _ in range(2):
        x = PixelConvLayer3(mask_type = "B")(x)

    out = keras.layers.Conv2D(
        filters=3, kernel_size=1, strides=1, activation="sigmoid", padding="valid"
    )(x)

    pixel_cnn = keras.Model(inputs, out)
    adam = keras.optimizers.Adam(learning_rate=0.0005)
    pixel_cnn.compile(optimizer=adam, loss="binary_crossentropy")

    pixel_cnn.summary()
    # pixel_cnn.fit(
    #     x=data, y=data, batch_size=128, epochs=50, validation_split=0.1, verbose=2
    # )
    
    pixel_cnn.fit(data, batch_size = 32, epochs = epochs, callbacks = [model_save_callback])
    
    pixel_cnn.save(model_folder +  "/model_final.h5")
    
    
    
if __name__ == '__main__':
    main()
