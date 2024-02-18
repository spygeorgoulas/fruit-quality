import argparse
import os
from skimage import io, img_as_ubyte
from skimage.transform import resize
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from ..helpers.ExtractFeatures import extract_all_features
from ..helpers.RemoveBackground import remove_background
from ..helpers.Augment import augment

def create_df(input_folder_path):
    images_list = []
    labels_list = []
        
    quality_list = os.listdir(input_folder_path)
        
    for quality in quality_list:
        fruit_type = os.listdir(os.path.join(input_folder_path, quality))
        for fruit in fruit_type:
            #if 'Banana' in fruit:
            fruit_path = os.path.join(input_folder_path, quality, fruit)
            images = os.listdir(fruit_path)
            for image in images:
                image_path = os.path.join(fruit_path, image)
                print(f"Reading: {image_path}")
                i = io.imread(image_path)
                if len(i.shape) == 2 or (len(i.shape) == 3 and i.shape[2] == 1):
                    #print(f"Grayscale {image_path}")
                    continue  
                images_list.append(image_path)
                # Check if the image name contains '_Good' or '_Bad' and assign the label accordingly
                if "_Good" in fruit or "_Bad" in fruit:
                    labels_list.append(f"{fruit}")
                else:
                    labels_list.append(f"{fruit}_Mixed")
                        
    images_series = pd.Series(images_list, name="images_paths")
    labels_series = pd.Series(labels_list, name="labels")
    return pd.concat([images_series, labels_series], axis=1)


def trim (df, max_size, min_size, column):
    df=df.copy()
    original_class_count= len(list(df[column].unique()))
    print ('Original Number of classes in dataframe: ', original_class_count)
    sample_list=[] 
    groups=df.groupby(column)
    for label in df[column].unique():        
        group=groups.get_group(label)
        sample_count=len(group)         
        if sample_count> max_size :
            strat=group[column]
            samples,_=train_test_split(group, train_size=max_size, shuffle=True, random_state=123, stratify=strat)            
            sample_list.append(samples)
        elif sample_count>= min_size:
            sample_list.append(group)
    df=pd.concat(sample_list, axis=0).reset_index(drop=True)
    final_class_count= len(list(df[column].unique())) 
    if final_class_count != original_class_count:
        print ('*** WARNING***  dataframe has a reduced number of classes' )
    balance=list(df[column].value_counts())
    print (balance)
    return df   

def extract_features(feature_save_path, input_df):
    resize_shape=(256, 192)
    for index, row in input_df.iterrows():
        # Load image
        image_path = row['images_paths']
        image = io.imread(image_path)
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            continue  
        image = remove_background(image)
        image = resize(image, resize_shape, anti_aliasing=True)
        image = img_as_ubyte(image)
    
        # Extract features
        features = extract_all_features(image)

        # Extract label
        label = row['labels']

        # Save features and label to a file
        data = {
            'features': features,
            'label': label
        }
        feature_file_path = Path(feature_save_path) / f'{Path(image_path).stem}_data.npy'
        np.save(feature_file_path, data)
        print(f'Data saved to: {feature_file_path}')


if __name__ == "__main__":
    # Define the argument parser to read in the input and output folder paths
    parser = argparse.ArgumentParser(description='Preprocess and extract features of given folder')
    parser.add_argument('-i', '--input_folder_path', type=str, required=True, help='Path to the dataset folder')

    # Parse the arguments
    args = parser.parse_args()
    
    df = create_df(args.input_folder_path)
    
    groups = df.groupby('labels')
    
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    # For each group, split its data into train and test, then append to train_df and test_df
    for label, group in groups:
        train, test = train_test_split(group, test_size=0.2, random_state=42)
        train_df = pd.concat([train_df, train])
        test_df = pd.concat([test_df, test])

    # Shuffle the datasets
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    # Checking the balance in the datasets
    print("Training set class distribution:")
    print(train_df['labels'].value_counts())

    print("\nTesting set class distribution:")
    print(test_df['labels'].value_counts())
    
    max_samples = 400
    min_samples = 0
    column = 'labels'
    train_df = trim(train_df, max_samples, min_samples, column)
    
    train_df = augment(train_df, 'output_augment', resize_shape=(256, 129))
    balance=list(train_df['labels'].value_counts())
    
    extract_features('features_train', train_df)
    extract_features('features_train', test_df)    