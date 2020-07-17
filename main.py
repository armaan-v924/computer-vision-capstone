# Imports
import Profile
import database_functions
import generating_descriptors
import camera
import display_image

# Main
function = input("Would you like to (1) Add a profile or (2) identify from a picture? ") # Identify what to do

try: 
    function = int(function) # Cast to int & check input
except ValueError: 
    print('Please enter only "1" or "2"')
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

else:
    # Match Faces and Display Boxes
    display_image.display_image()
    pass