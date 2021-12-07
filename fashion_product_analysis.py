import os
import sys

import numpy as np
import pandas as pd
import shutil
import json

import matplotlib.pyplot as plt

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)


def copy_subdata():
    data_folder = base_folder + '/data/fashion_product/fashion-dataset'
    
    styles_file = data_folder + '/styles.csv'
    print(styles_file)
    styles = pd.read_csv(styles_file)
    
    print(styles)
    
    ### ['Apparel' 'Accessories' 'Footwear' 'Personal Care' 'Free Items' 'Sporting Goods' 'Home']
    print(styles['masterCategory'].unique())
    
    #style_subset = styles[styles['masterCategory'] == 'Apparel'][styles['subCategory'] == 'Topwear'][styles['gender'] == 'Men']
    style_subset = styles[styles['masterCategory'] == 'Apparel'][styles['subCategory'] == 'Topwear'][styles['gender'] == 'Women']
    print(style_subset)
    
    sys.exit(0)
    
    source_folder = data_folder + '/images'
    #destination_folder = data_folder + '/images_subfolder/apparel/men'
    destination_folder = data_folder + '/images_subfolder/apparel/women'
    
    #destination_folder = data_folder + '/images_subfolder/test'
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for index, row in style_subset.iterrows():
        img_id = row['id']
        print(img_id)
        
        # if (index > 100):
        #     break
        
        img_file = '{0}.jpg'.format(img_id)
        source_path = source_folder + '/' + img_file
        destination_path = destination_folder + '/' + img_file
        
        if (not os.path.exists(source_path)):
            continue
            
        if (os.path.exists(destination_path)):
            continue
            
        shutil.copyfile(source_path, destination_path)
        
        #sys.exit(0)
        
def img_style():
    image_style_folder = base_folder + '/data/fashion_product/fashion-dataset/styles'
    img_id = 1163
    img_style_file = '{0}.json'.format(img_id)
    
    
    # print(os.listdir(image_style_folder))
    # sys.exit(0)
    
    style_info_all = []
    for img_style_file in os.listdir(image_style_folder):
        fp = open(image_style_folder + '/' + img_style_file, 'r')
        img_style = json.load(fp)
        fp.close()
        
        print(img_style)
        print(img_style.keys())
        
        print(json.dumps(img_style, indent = 4))
        
        sys.exit(0)
        
        # print('-------------------------')
        # print(img_style['data'].keys())
        
        style_info = [
            img_style_file,
            img_style['data']['styleType'],
            img_style['data']['myntraRating'],
            img_style['data']['ageGroup'],
            img_style['data']['gender'],
            img_style['data']['baseColour'],
            img_style['data']['fashionType'],
            img_style['data']['articleAttributes'].get('Fit'),
            img_style['data']['masterCategory'].get('typeName'),
            img_style['data']['subCategory'].get('typeName'),
            img_style['data']['articleType'].get('id'),
            img_style['data']['articleType'].get('typeName')
        ]
        
        print(style_info)
        
        style_info_all.append(style_info)
        
    columns = ['img_f', 'style_type', 'rating', 'age_group', 'gender', 'base_color', 
                    'fashion_type', 'fit', 'master_category', 'sub_category', 'article_type_id', 'article_type_name']
                    
    style_info_all = pd.DataFrame(np.array(style_info_all), columns = columns)
    print(style_info_all)
    
    #style_info_all.to_csv(base_folder + '/data/fashion_product/fashion-dataset/styles_all.csv', index = None)
    
def main():
    print(5)
    copy_subdata()
    #img_style()
    
if __name__ == '__main__':
    main()
