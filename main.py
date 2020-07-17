# Imports
import database_functions
import camera
import display_image
import whisper as wsp
from pathlib import Path
import database_functions

# Main
function = input("Would you like to (1) Add a profile or (2) identify from a picture or (3) add multiple profiles? ") # Identify what to do

try: 
    function = int(function) # Cast to int & check input
except ValueError: 
    print('Please enter only "1" or "2" or "3"')
except:
    print('Something went wrong. Please try again.') # Probably shouldn't run, but just in case I miss something


if function == 1:
    # Add a profile
    imput = input('(1) Image file\n(2) Take Picture ')

    try: 
        imput = int(imput) # Cast to int & check input
    except ValueError: 
        print('Please enter only "1" or "2"')
    except:
        print('Something went wrong. Please try again.') # Probably shouldn't run, but just in case I miss something

    # Add person to database
    database_functions.add_image(camera.take_picture() if imput == 2 else input("Please enter the image file path"), input("What is this person's name? "))

elif function == 2:
    # Match Faces and Display Boxes
    display_image.display_image()
    pass

elif function == 3:
    print("Please ensure that all of and only the pictures you would like to add are stored in a single folder.\n")
    path = input("Enter the relative file location to your pictures folder (not in quotes): ")
    clusters, graph = wsp.whisper_img(path)
    wsp.cluster_to_profile(clusters, graph)
