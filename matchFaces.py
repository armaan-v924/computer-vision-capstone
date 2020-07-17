from Profile import Profile
import numpy as np

def cos_distance(d1, d2):
    '''
    Computes the cosine distance between face descriptors

    Parameters:
    -----------
    d1: facial descriptor for first image, numpy array
    d2: facial descriptor for second image, numpy array

    Returns:
    --------
    A number from [0,2] representing the cosine distance of d1 and d2
    '''
    return 1- np.dot(d1,d2)/(np.sqrt(np.dot(d2,d2)) * np.sqrt(np.dot(d2,d2)))

def match_face(descriptor, database, threshold):
    '''
    Compares a given descriptor with the mean descriptor of each profile

    Parameters:
    -----------
    descriptor: given descriptor to compare
    database: database holding the profiles
    threshold: value to gauge the the cos_distance between descriptors

    Returns: name of a potential match or "No match" if there are none
    --------

    '''
    potential_matches = []
    for name,profile in database.items():
        dist = cos_distance(descriptor, profile.compute_mean()) #change
        if dist < threshold:
            potential_matches.append(name)
    if len(potential_matches) > 0:
        return potential_matches[0]
    return "No matches"
