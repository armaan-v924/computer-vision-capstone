"""Functionality to display an image with a box around detected faces with labels to indicate matches or an “Unknown” label otherwise"""
import generating_descriptors as gd
import matplotlib.pyplot as plt
from facenet_models import FacenetModel
from camera import take_picture
from matplotlib.patches import Rectangle
import matchFaces as mf
import database_functions as df
def display_image():
    img_array = take_picture()
    descriptors, bounding_boxes, probabilities, landmarks = gd.find_faces(img_array)
    fig,ax = plt.subplots()
    ax.imshow(img_array)
    for descriptor, box, prob, _ in zip(descriptors, bounding_boxes, probabilities, landmarks):
        #draws the box on the plot
        ax.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=None, lw=2, color="purple"))
        label = mf.match_face(descriptor, df.load_db("database.pkl"),500)
        ax.text(box[:2], *(box[2:] - box[:2]), label, fontsize=15)
    plt.show()



