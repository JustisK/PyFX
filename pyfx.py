import os, sys, re
import argparse
import numpy as np
import h5py
import skimage.io
import skimage.transform
from sklearn.preprocessing import LabelBinarizer
from keras.applications import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Reshape

# Command line arguments: image in-path, feature out-path, extension for output
parser = argparse.ArgumentParser(description='Perform InceptionV3-ImageNet feature extraction on images.')

"""parser.add_argument('strings', metavar='img_path, out_path, or extension', type=str, nargs='+',
	                help='a string (either directory or file extension)', action='append')
"""

parser.add_argument(nargs='?', type=str, dest='img_path',
                    default='./images', action='store')
parser.add_argument(nargs='?', type=str, dest='out_path',
                    default='./output/features', action='store')
#parser.add_argument(nargs='?', type=str, dest='ext',
                    #default='csv', action='store')
args = parser.parse_args()

def extract_features():

    # Load dataset (any image-based dataset)
    matches = [(re.match(r'^(([a-zA-Z]+)\d+\.png)', fname), path) 
    		for path, dirs, files in os.walk(''+str(args.img_path)) 
    		for fname in files]
    patches = [skimage.transform.resize( # resize image to (256, 256)
    	skimage.io.imread(os.path.join(path, match.group(1))), # open each image
	    (256, 256)) for match, path in matches if match]
    
    # Preprocess for InceptionV3	
    patches = preprocess_input(np.array(patches))

    # Construct model (using ImageNet weights)
    inceptionV3 = InceptionV3(weights = "imagenet", include_top = False, 
        input_shape = patches[0].shape)
    
    # Isolate pre-softmax outputs
    x = inceptionV3.output

    """
    # Fluff
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
	"""

    # Construct extractor model
    extractor = Model(inputs=[inceptionV3.input], outputs=[x])

    # Extract features with Model.predict()
    features = extractor.predict(x=patches, batch_size=2)

    # TODO: get rid of zero-padding
    # TODO: concatenate individual patches to individual 1d arrays?

    return features

features = extract_features()
# save by pickling (not recommended):
# np.save(file=str(args.out_path), allow_pickle=True, arr=features)

# (Recommended) save --> .hdf
f = h5py.File(""+str(args.out_path)+".hdf5", "w")
hdf = f.create_dataset(name=str(args.out_path), data=features)

import gc; gc.collect()