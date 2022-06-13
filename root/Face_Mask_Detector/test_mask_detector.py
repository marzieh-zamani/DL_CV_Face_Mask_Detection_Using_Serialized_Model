# USAGE
# python test_mask_detector1.py --dataset mz_dataset

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import matplotlib.pyplot as plt
import pdb #mz  pdb.set_trace() #mz

# initialize the initial learning rate, number of epochs to train for, and batch size
BS = 32

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
modelAW = load_model("mask_detector_AW.model")
modelAWP = load_model("mask_detector_AWP.model")
modelRWP = load_model("mask_detector_RWP.model")
lb = LabelBinarizer()

# load datasets
datasetPaths1=['dataset/dataset_test/ds1_RW', 'dataset/dataset_test/ds1_RP']

datasetPaths2=['dataset/dataset_test/ds2_RW', 'dataset/dataset_test/ds2_RP']


res=np.zeros([3,2])
for ind, dataset_path in enumerate(datasetPaths1):

    # grab the list of images in our dataset directory, then initialize
    # the list of data (i.e., images) and class images
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(dataset_path))
    data = []
    labels = []

    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        #pdb.set_trace()
        label = imagePath.split(os.path.sep)[-2]
        # load the input image (224x224) and preprocess it
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)

    # convert the data and labels to NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    #pdb.set_trace() #mz
    # perform one-hot encoding on the labels
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    testX = data
    testY = labels

    #  ********** MODEL A-W ********** 
    # make predictions on the testing set
    print("[INFO] evaluating network...")
    predIdxs = modelAW.predict(testX, batch_size=BS)

    # for each image in the testing set we need to find the index of the label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)

    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))
    #pdb.set_trace()
    rep=classification_report(testY.argmax(axis=1), predIdxs,
target_names=lb.classes_, output_dict=True)
    res[0,ind]=rep['macro avg']['precision']

    # ********** MODEL A-WP ********** 
    # make predictions on the testing set
    print("[INFO] evaluating network...")
    predIdxs = modelAWP.predict(testX, batch_size=BS)

    # for each image in the testing set we need to find the index of the label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)

    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

    rep=classification_report(testY.argmax(axis=1), predIdxs,
target_names=lb.classes_, output_dict=True)
    #pdb.set_trace()
    res[1,ind]=rep['macro avg']['precision']

for ind, dataset_path in enumerate(datasetPaths2):

    # grab the list of images in our dataset directory, then initialize
    # the list of data (i.e., images) and class images
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(dataset_path))
    data = []
    labels = []

    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        #pdb.set_trace()
        label = imagePath.split(os.path.sep)[-2]
        # load the input image (224x224) and preprocess it
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)

    # convert the data and labels to NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    #pdb.set_trace() #mz
    # perform one-hot encoding on the labels
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels

    testX = data
    testY = labels

    # ********** MODEL R-WP ********** 
    # make predictions on the testing set
    print("[INFO] evaluating network...")
    predIdxs = modelRWP.predict(testX, batch_size=BS)

    # for each image in the testing set we need to find the index of the label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)

    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

    rep=classification_report(testY.argmax(axis=1), predIdxs,
target_names=lb.classes_, output_dict=True)

    res[2,ind]=rep['macro avg']['precision']


pdb.set_trace()

prec=res
# ****** Figure 1 ******
plt.style.use("ggplot");
plt.figure();
plt.bar(np.arange(0, prec.shape[1]),prec[0,:], width=-0.2, align='edge', label='Model A_W');
plt.bar(np.arange(0, prec.shape[1]),prec[1,:], width=0.2, align='edge', label='Model A_WP');
plt.legend(loc='best');
plt.xticks(np.arange(0, prec.shape[1]),['Ds_Real_White','Ds_Real_Pattern']);
plt.title("Face Mask Detection \n Trained on Art. Dataset | Tested on Real Dataset");
plt.xlabel("Dataset");
plt.ylabel("Classification Precision");
plt.savefig('res/fig1.png');


# ****** Figure 1 ******
plt.style.use("ggplot");
plt.figure();
plt.bar(np.arange(0, prec.shape[1]),prec[1,:], width=-0.2, align='edge', label='Model A_WP');
plt.bar(np.arange(0, prec.shape[1]),prec[2,:], width=0.2, align='edge', label='Model R_WP');

plt.legend(loc='best');
plt.xticks(np.arange(0, prec.shape[1]),['Ds_Real_White','Ds_Real_Pattern']);
plt.title("Face Mask Detection \n Trained on Art.&Real Dataset | Tested on Real Dataset");
plt.xlabel("Dataset");
plt.ylabel("Classification Precision");
plt.savefig('res/fig2.png');

pdb.set_trace()


