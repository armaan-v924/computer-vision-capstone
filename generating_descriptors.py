# Imports
from facenet_models import FacenetModel
import torch
import numpy as np
import matplotlib as plt
from matplotlib.image import imread
import cv2

# Create Rectangles
def find_faces(image):
    """ Using facenet_models, locate faces in a given picture and create descriptor vecors

    Parameters:
    -----------
    image: Path to image file
    
    Returns:
    --------
    Tuple: (descriptor: (N, 512) array, N is number of faces, 
            bounding_box: (N, 4) array, bounds of each face in the image,
            probabilities: (N,) array, confidence (0-1))

    """
    # Format image
    img = cv2.imread(image)
    img = img[:,:,::-1]

    # Create model
    model = FacenetModel()

    # Detect Faces
    bounding_boxes, probabilities, landmarks = model.detect(img)

    # Create descriptors
    descriptors = model.compute_descriptors(img, bounding_boxes)

    return descriptors, bounding_boxes, probabilities

# 