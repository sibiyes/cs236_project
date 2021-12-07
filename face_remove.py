import os
import sys
import numpy as np

import mtcnn

import matplotlib.pyplot as plt

# https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

def remove_face(detector, image):
    # detect faces in the image
    face_extract = detector.detect_faces(image)
    
    if (len(face_extract) > 0):
    
        # extract the bounding box from the first face
        x1, y1, width, height = face_extract[0]['box']
        x2, y2 = x1 + width, y1 + height
        
        # extract the face
        face = np.copy(image[y1:y2, x1:x2])
        
        mask_buffer = int(0.3*(y2-y1))
        image_noface = np.copy(image[y2+mask_buffer:-1, :])
        
        return image_noface
    
    else:
        
        return image

def main():
    print(5)
    
    # input_image_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder/apparel/men'
    # output_image_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder_clean/apparel/men'
    
    input_image_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder/apparel/women'
    output_image_folder = base_folder + '/data/fashion_product/fashion-dataset/images_subfolder_clean/apparel/women'
    
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
        
    for img_f in os.listdir(input_image_folder):
        print(img_f)
    
        image = plt.imread(input_image_folder + '/' + img_f)
        detector = mtcnn.MTCNN()
        
        image_noface = remove_face(detector, image)
        
        plt.imsave(output_image_folder + '/' + img_f, image_noface)
    

    
if __name__ == '__main__':
    main()
