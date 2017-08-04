import os, sys, re
import argparse
import numpy as np
import skimage.io
import skimage.transform
from sklearn.preprocessing import LabelBinarizer
from keras.applications import InceptionV3
from keras.applications import imagenet_utils
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
parser.add_argument(nargs='?', type=str, dest='ext',
                    default='csv', action='store')
args = parser.parse_args()

def extract_features():

    # Load dataset (any image-based dataset)
    matches = [(re.match(r'^(([a-zA-Z]+)\d+\.png)', fname), path) 
	    for path, dirs, files in os.walk(''+str(args.img_path)) for fname in files]
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

    x = inceptionV3.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    extractor = Model(inputs=[inceptionV3.input],outputs=[x])
    features = extractor.predict(x=patches, batch_size=2)
    return features

features = extract_features()
# print("\n[INFO] Output array shape:", features.shape) # Uncomment to verify output shape
np.savetxt("" + str(args.out_path) + "." + str(args.ext), features, fmt='%f')
