import whisper as wsp
from pathlib import Path
import database_functions

# Main
root = Path(".")
print("Developed by @therealshazam\n")
print("Please ensure that all of and only the pictures you would like to add are stored in a single folder.\n")
path = input("Enter the relative file location to your pictures folder (not in quotes): ")
clusters, graph = wsp.whisper_img(path)
wsp.cluster_to_profile(clusters, graph, "file.pck")
# except:
#     print("Sorry something went wrong.")
