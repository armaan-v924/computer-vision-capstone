"""Create a Profile class with functionality to add face descriptors and compute the mean descriptor"""
"""__init___ = Creates an empty list that contains face descriptors. This list will constantly be updated and saved everytime a new face descriptor was added.""" 
import numpy as np
class Profile: 
    def __init__(self, profile_name):  
        self.face_list = []
        self.name = profile_name
    def add_face_descriptor(self, face_descriptor):
        self.face_list.append(face_descriptor)
    
    def compute_mean(self):
        mean_descriptor = 0.0
        for fd in self.face_list:
            mean_descriptor += fd
        return mean_descriptor / len(self.face_list)

    


