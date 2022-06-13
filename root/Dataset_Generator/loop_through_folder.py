# USAGE
# python loop_through_folder.py

from imutils import paths
import cv2
import os
from mask import create_mask
import pdb #mz pdb.set_trace() #mz

# grab the list of transparent masks in our dataset directory
maskPaths = list(paths.list_images('dataset/PetternedMasks'))

folder_path = "dataset/faces"

images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for i in range(len(images)):
    print("the path of the image is", images[i])
    mask_path=maskPaths[i%7]
    create_mask(images[i], mask_path)
