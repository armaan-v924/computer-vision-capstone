
import pickle
import generating_descriptors as gd
import Profile

def load_db(pathname):
    """
    returns the stored database from a pickle file
    
    Parameters
    ----------
    pathname: string
    
    Returns
    -------
    database: dictionary mapping names to profiles
    
    """
    with open(pathname, mode="rb") as opened_file:
        database = pickle.load(opened_file)
    return database
    
def save_db(database, pathname):
    """
    saves the given database into a pickle file
    
    Parameters
    ----------
    database: dictionary
    pathname: string
    
    """
    with open(pathname, mode="wb") as opened_file:
        pickle.dump(database, opened_file)

def add_profile(profile):
    """
    adds a new profile to the database {profile.name: profile}
    
    Parameters
    ----------
    profile: Profile of the person to add
    
    """
    database = load_db("database.pkl")
    database[profile.name] = profile
    save_db(database, "database.pkl")
    
def remove_profile(profile):
    """
    removes a profile from the database
    
    Parameters
    ----------
    profile: Profile of the person to remove
    
    """
    database = load_db("database.pkl")
    database.pop(profile.name)
    save_db(database, "database.pkl")
    
    
def add_image(img, name):
    """
    adds the descriptor of the image to the correct profile in the database
    
    Parameters
    ----------
    img: string - pathname of image
    name: string - name of person to add to
    
    """
    descriptor = gd.find_faces(img)[0]
    database = load_db("database.pkl")
    
    if not name in database:
        database[name] = Profile.Profile(name)
    
    database[name].add_face_descriptor(descriptor)
    save_db(database, "database.pkl")
    

