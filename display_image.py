"""Functionality to display an image with a box around detected faces with labels to indicate matches or an “Unknown” label otherwise"""
import generating_descriptors as gd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from camera import take_picture

img_array = take_picture()
descriptors, bounding_boxes, probs = gd.find_faces(img_array)
img.plot(img_array)




