# USAGE
# python patterned_masks.py

from imutils import paths
import cv2
import os
import pdb #mz  pdb.set_trace() #mz

# read transparent mask with "RGBA" format
mask=cv2.imread("dataset/images/blue-mask.png", cv2.IMREAD_UNCHANGED)

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images('dataset/Patterns'))

# loop over the image paths
for imagePath in imagePaths:


	# load the input image (224x224) and preprocess it
	pat = cv2.imread(imagePath)
	pat = cv2.resize(pat, mask.shape[1::-1])
	rgb = cv2.bitwise_and(pat, mask[:,:,0:3])
	rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
	rgba[:, :, 3] = mask[:,:,3]

	# extract the filename and save in new folder
	maskPath = 'dataset/PetternedMasks/' + imagePath.split(os.path.sep)[-1]
	cv2.imwrite(maskPath, rgba)

