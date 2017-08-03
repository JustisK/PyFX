import os, sys, re
import numpy as np
import skimage.io
import skimage.transform
from sklearn.preprocessing import LabelBinarizer
from keras.applications import InceptionV3
from keras.applications import imagenet_utils
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Reshape

def extract_features():

    # Load dataset (any image-based dataset)
    matches = [(re.match(r'^(([a-zA-Z]+)\d+\.png)', fname), path) 
	    for path, dirs, files in os.walk('./datasets/ucm') for fname in files] # TODO argv directory
    patches = [skimage.transform.resize( # resize image to (256, 256) TODO argv patch size?
	    skimage.io.imread(os.path.join(path, match.group(1))), # open each image
    	    (256, 256)) for match, path in matches if match]
	
    # change color channel order and shift mean color for ImageNet		
    patches = imagenet_utils.preprocess_input(np.array(patches)) 
    labels = [match.group(2) for match, path in matches if match]
    labels = LabelBinarizer().fit_transform(list(labels)) # one-hot encoding
    print('patches', patches[0].shape, len(patches), 'labels', len(labels))

    # Construct model (using ImageNet weights)
    inceptionV3 = InceptionV3(weights = "imagenet", include_top = False, 
        input_shape = patches[0].shape)
	
    #for layer in inceptionV3.layers:
    #    layer.trainable = False

    x = inceptionV3.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    extractor = Model(inputs=[inceptionV3.input],outputs=[x])
    features = extractor.predict(x=patches, batch_size=2)
    return features

features = extract_features()
# print("\n[INFO] Output array shape:", features.shape)
np.savetxt('features.csv', features, fmt='%f') # TODO argv filename
