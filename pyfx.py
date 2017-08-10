import os, re
import argparse
import numpy as np
import h5py
import skimage.io
import skimage.transform
from keras.applications import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dropout, Flatten, Input

# Command line arguments: image in-path, feature out-path, extension for output
parser = argparse.ArgumentParser(description='Perform InceptionV3-ImageNet feature extraction on images.')

parser.add_argument(nargs='?', type=str, dest='img_path',
                    default='./images', action='store')
parser.add_argument(nargs='?', type=str, dest='out_path',
                    default='./output/features', action='store')
parser.add_argument(nargs='?', type=str, dest='ext',
                    default='hdf5', action='store')
parser.add_argument(nargs='?', type=bool, dest='compressed',
                    default=True, action='store')
args = parser.parse_args()

"""
Extracts feature data for each member in a directory containing .png images,
then stores them in a user-selected file format (by default, hdf5).
"""

def extract_multi():
    # Load dataset (any image-based dataset)
    matches = [(re.match(r'^(([a-zA-Z]+)\d+\.png)', fname), path)
               for path, dirs, files in os.walk('' + str(args.img_path))
               for fname in files]
    patches = [skimage.transform.resize(  # resize image to (256, 256)
        skimage.io.imread(os.path.join(path, match.group(1))),  # open each image
        (256, 256)) for match, path in matches if match]

    # Preprocess for InceptionV3	
    patches = preprocess_input(np.array(patches))

    # Construct model (using ImageNet weights)
    inceptionV3 = InceptionV3(weights="imagenet", include_top=False,
                              input_shape=patches[0].shape)

    # Isolate pre-softmax outputs
    x = inceptionV3.output

    # Experimental - flatten to 2d for CSV
    if args.ext == "csv":
        x = Flatten()(x)
        x = Dropout(0.5)(x)

    # Construct extractor model
    extractor = Model(inputs=[inceptionV3.input], outputs=[x])

    # Extract features with Model.predict()
    features = extractor.predict(x=patches, batch_size=2)

    # TODO: get rid of zero-padding
    # TODO: concatenate individual patches to individual 1d arrays?

    return features


"""
Returns feature data for a single image or patch. Intended as a helper
method for an extract_multi() variant that returns 1d arrays of feature
data for each member in a list of images - but, can be used explicitly.

Those intending to use this method directly might consider libkeras's
extract_features.py as an alternative.
"""


def extract_single():
    target = image.load_img(args.img_path)


def save_features():
    # HARD CODED TEST STUFF BELOW
    features = extract_multi()
    print(features.shape)
    extension = str(args.ext)
    if extension == "hdf5":
        # (Recommended, default) save --> .hdf
        f = h5py.File("" + str(args.out_path) + ".hdf5", "w")
        hdf = f.create_dataset(name=str(args.out_path), data=features)
    elif extension == "npy":  # god please don't actually do this
        outfile = "" + str(args.out_path)
        np.save(file=outfile, allow_pickle=True, arr=features)
    else:
        # TODO: add Flatten()(features) layer --> reduce to 2d array
        if args.compressed:
            extension += ".gz"
        outfile = "" + str(args.out_path) + "." + extension
        np.savetxt(fname=outfile, X=features, fmt='%1.6f')
    # TODO: npz for the optional list of concat. 1d arrays
    # END HARD CODED TEST STUFF

save_features()
import gc; gc.collect()
