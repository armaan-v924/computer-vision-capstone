# Imports
from facenet_models import FacenetModel
import torch
import numpy as np
from matplotlib.image import imread
import cv2

# Create Rectangles
def find_faces(image):
    """ Using facenet_models, locate faces in a given picture and create descriptor vecors

    Parameters:
    -----------
    image: Path to image file OR (X, Y, 3) numpy array of pixels
    
    Returns:
    --------
    Tuple: (descriptor: (N, 512) array, N is number of faces, 
            bounding_box: (N, 4) array, bounds of each face in the image,
            probabilities: (N,) array, confidence (0-1))

    """
    # Format image
    if type(image).__module__ is not np.__name__:
        img = cv2.imread(image)
        img = img[:,:,::-1]
    else:
        img = image

    # Create model
    model = FacenetModel()

    # Detect Faces
    bounding_boxes, probabilities, landmarks = model.detect(img)

    # Create descriptors
    descriptors = model.compute_descriptors(img, bounding_boxes)

    return descriptors, bounding_boxes, probabilities

# 