import os, sys, re
import numpy as np
import pandas as pd
import skimage.io
import skimage.transform
from sklearn.preprocessing import LabelBinarizer
from keras.applications import InceptionV3
from keras.applications import imagenet_utils
from keras.models import Model
from keras.layers import Input, Dense, Output, Flatten, Dropout

def extract_features():

    # Load dataset (any image-based dataset)
    matches = [(re.match(r'^(([a-zA-Z]+)\d+\.png)', fname), path) 
	    for path, dirs, files in os.walk('../') for fname in files]
    patches = [skimage.transform.resize( # resize image to (256, 256) TODO argv param based on patch size
	    skimage.io.imread(os.path.join(path, match.group(1))), # open each image
    	    (256, 256)) for match, path in matches if match]
	
    # change color channel order and shift mean color for ImageNet		
    patches = imagenet_utils.preprocess_input(np.array(patches)) 
    labels = [match.group(2) for match, path in matches if match]
    labels = LabelBinarizer().fit_transform(list(labels)) # one-hot encoding
    print('patches', patches[0].shape, len(patches), 'labels', len(labels))

    # Construct model (using ImageNet weights)
    self.inceptionV3 = InceptionV3(weights = "imagenet", include_top = False, 
        pooling = avg, input_shape = patches[0].shape)
	
    for layer in inceptionV3.layers:
	    layer.trainable = False

    x = inceptionV3.outputs
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    
    # actual feature extraction
    features = inceptionV3.predict(x, batch_size=1)
