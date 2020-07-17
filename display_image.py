"""Functionality to display an image with a box around detected faces with labels to indicate matches or an “Unknown” label otherwise"""
import generating_descriptors as gd
import matplotlib.pyplot as plt
from facenet_models import FacenetModel
from camera import take_picture
from matplotlib.patches import Rectangle
import matchFaces as mf
import database_functions as df
import cv2
def display_image():
    """Using camera.take_picture() and mf.match_face, plot image and display boxes & names around faces

    Parameters:
    -----------
    None

    Returns:
    --------
    None 
    """
    cam = int(input("Take a picture (1) or use a file (2)? "))
    if cam == 1:
        img_array = take_picture()
    else:
        img = cv2.imread(input("Enter your file path: "))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = img
    descriptors, bounding_boxes, probabilities, landmarks = gd.find_faces(img_array)
    _,ax = plt.subplots()
    ax.imshow(img_array)
    for descriptor, box, _, _ in zip(descriptors, bounding_boxes, probabilities, landmarks):
        #draws the box on the plot
        ax.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=None, lw=2, color="purple"))
        label = mf.match_face(descriptor.reshape(512, 1), df.load_db("database.pkl"),0.7)
        ax.text(box[0],box[1]-10, label, fontsize=8,bbox={'facecolor': 'purple','alpha': 0.5, 'pad': 5})
    plt.show()



