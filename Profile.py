import numpy as np
class Profile: 
    """The Profile class serves as a blueprint for all face objects. It has three functions, the initalize function, a function to add 
    face descriptors to the list, and another function that computes the mean descriptor. This is very necessary for identifying faces since it is
    needed to create a database that is used for comparing different faces and saving new faces.
    """
    def __init__(self, profile_name):  
        """Initializes the face object by creating an empty face_list and initializing its mean (self.mean). 
        It also passes profile_name to self.name.
        
        Parameters
        ----------
        profile_name ([String]): highlights the name of a profile. This is going to be saved onto self.name, a parameter of the class.
        
        Returns:
        --------
        None
        """
        self.face_list = []
        self.name = profile_name
        self.mean = 0.0
    def add_face_descriptor(self, face_descriptor):
        """Adds the given face_descriptor to the list of descriptors (self.face_list). Computes the mean of face_list immediately after 
        the face_descriptor is added.

        Parameters:
        -----------
        face_descriptor ([np.array]): an array full of unique data that highlights a face from that of others.
        
        Returns:
        --------
        None
        """
        self.face_list.append(face_descriptor)
        self.mean = self.compute_mean()
    
    def compute_mean(self):
        """Takes the mean of all face_descriptors that are on face_list. It does so by adding all face_descriptors to a temporary 
        variable called mean_descriptor. Finally, it returns the sum divided by the number of elements in face_list

        Parameters:
        -----------
        None

        Returns:
        --------
        mean[float]: the average of the elements on face_list.
        """
        mean_descriptor = 0.0
        for fd in self.face_list:
            mean_descriptor += fd
        return mean_descriptor / len(self.face_list)

    


