from skimage import io, img_as_ubyte
from skimage.transform import resize, rotate
from skimage.exposure import adjust_gamma
from pathlib import Path
import pandas as pd

def augment_image(image, output_path, image_name, transformations, file_extension='.jpg'):
    augmented_images = []
    # Apply each transformation and save the result
    for trans_name, trans_func in transformations.items():
        transformed_image = trans_func(image)
        transformed_image = img_as_ubyte(transformed_image)  # Convert to uint8
        augmented_image_name = f'{trans_name}_{image_name}{file_extension}'
        augmented_image_path = output_path / augmented_image_name
        io.imsave(augmented_image_path, transformed_image)
        print(f'Augmented image saved to: {augmented_image_path}')
        augmented_images.append(augmented_image_path)
    return augmented_images

def augment(df, output_path, resize_shape):
    # Define transformations
    transformations = {
        'rotated_90': lambda x: rotate(x, 90),
        'rotated_180': lambda x: rotate(x, 180),
        'rotated_270': lambda x: rotate(x, 270),
        'gamma_0.8': lambda x: adjust_gamma(x, 0.8),
        'gamma_1.2': lambda x: adjust_gamma(x, 1.2)
    }

    # Calculate max count for each label and group by label
    max_count = 500
    grouped = df.groupby('labels')

    new_rows = []  # To store new augmented image paths and labels

    for label in df['labels'].unique():
        # Determine how many images to augment for this label
        group = grouped.get_group(label)
        
        augment_count = max_count - len(group)
        print(f"{label} length: {len(group)}")

        while augment_count > 0:
            for _, row in group.iterrows():
                if augment_count <= 0:
                        break

                image_path = row['images_paths']
                image_name = label + "_" +Path(image_path).stem

                # Load and preprocess image
                image = io.imread(image_path)
                if resize_shape is not None:
                    image = resize(image, resize_shape, anti_aliasing=True)
                    print(f"Images Resized {resize_shape}")
                image = img_as_ubyte(image)  # Convert to uint8

                # Perform augmentation and get augmented image paths
                augmented_image_paths = augment_image(image, Path(output_path), image_name, transformations)
                
                # Update DataFrame with new rows
                for aug_path in augmented_image_paths:
                    augment_count -= 1
                    if augment_count <= 0:
                        new_rows.append({'images_paths': str(aug_path), 'labels': label})
                        break
                    new_rows.append({'images_paths': str(aug_path), 'labels': label})
                    

    # Append new rows to the original DataFrame
    new_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df], ignore_index=True)
    return df