import cv2
import numpy as np

def get_optimized_rectangle(width, height):
    # Define margins as a percentage of the image's dimensions
    margin_x = int(width * 0.1) 
    margin_y = int(height * 0.1) 

    # Calculate rectangle coordinates
    x = margin_x
    y = margin_y
    rect_width = width - 2 * margin_x
    rect_height = height - 2 * margin_y

    return (x, y, rect_width, rect_height)

def remove_background(image):
    height, width = image.shape[:2]
    mask = np.zeros(image.shape[:2], np.uint8)
    rect = get_optimized_rectangle(width, height)

    # Allocate space for two arrays used by the GrabCut algorithm
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    # Apply the GrabCut algorithm
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    result = image * mask[:, :, np.newaxis]

    return result