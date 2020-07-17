"""Functionality to display an image with a box around detected faces with labels to indicate matches or an “Unknown” label otherwise"""
import generating_descriptors as gd
import matplotlib.pyplot as plt
from facenet_models import FacenetModel
from camera import take_picture

img_array = take_picture()
model = FacenetModel()
descriptors, bounding_boxes, probabilities = gd.find_faces(img_array)
fig,ax = plt.subplots()
ax.imshow(img_array)



