import os, re, gc, argparse
import numpy as np
import h5py
import skimage.io
import skimage.transform
from sklearn.feature_extraction.image import extract_patches_2d
from keras.applications import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dropout, Flatten, Input


def collect_args():
    """
    Collects command line arguments from invocation.
    :return: argparse object containing parsed command line arguments
    """
    # Command line arguments: image in-path, feature out-path, extension for output
    parser = argparse.ArgumentParser(description='Perform InceptionV3-ImageNet feature extraction on images.')

    parser.add_argument(nargs='?', type=str, dest='img_path',
                        default='./images', action='store')
    parser.add_argument(nargs='?', type=str, dest='out_path',
                        default='./output/features', action='store')
    parser.add_argument(nargs='?', type=str, dest='ext',
                        default='hdf5', action='store')
    parser.add_argument(nargs='?', type=bool, dest='compressed',
                        default=False, action='store')
    argv = parser.parse_args()
    compressed = argv.compressed
    extension = argv.ext
    # TODO: put this warning somewhere else
    if extension != ("csv" or "txt"):
        print("""WARNING: non-text output (bin, npy, hdf5) is incompressible for now.
        \nOutput will not be compressed.""")
    elif not compressed:
        print("""WARNING: non-compressed csv output is extremely large.
        Recommend re-running with compressed=True.""")
    return argv


def extract_multi():
    """
    extract_multi

    Extracts feature data for each member in a directory containing .png images.
    
    :return: Keras tensor containing extracted features.
    """

    # Load dataset (any image-based dataset)
    # TODO: argv for file types other than png
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

    # Construct extractor model
    extractor = Model(inputs=[inceptionV3.input], outputs=[x])

    # Extract features with Model.predict()
    features = extractor.predict(x=patches, batch_size=2)

    # TODO: get rid of zero-padding

    return features


def extract_single():
    """
    extract_single

    Returns feature data for a single image or patch. Does not concatenate
    output to a 1d array, but instead outputs a full Keras tensor. The 
    extraction is identical to extract_multi, but takes features from a 
    single file rather than a directory of files.
    
    Those intending to use this method directly might consider libkeras's
    extract_features.py as an alternative.
    
    :return: Keras tensor containing extracted features.
    """

    # Load dataset (any image-based dataset)
    target = image.load_image(args.img_path)
    patches = extract_patches_2d(target)

    # TODO: patch target --> patches[]
    """
    patches = [skimage.transform.resize(  # resize image to (256, 256)
        skimage.io.imread(os.path.join(path, match.group(1))),  # open each image
        (256, 256)) for match, path in matches if match]
    """
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

    # Construct extractor model
    extractor = Model(inputs=[inceptionV3.input], outputs=[x])

    # Extract features with Model.predict()
    features = extractor.predict(x=patches, batch_size=2)
    # TODO: get rid of zero-padding

    return features


def extract_single_1d():
    """
    extract_single_1d

    Returns feature data for a single image or patch. Intended as a helper
    method for an extract_multi() variant that returns 1d arrays of feature
    data for each member in a list of images - but, can be used explicitly.

    :return: Numpy array of features, concatenated to one dimension.
    """
    target = image.load_img(args.img_path)


def save_features(*kwargs):
    """
    Writes extracted feature vectors into a binary or text file, per args.
    :return: 
    """
    features = extract_multi()
    print(features.shape)  # comment out if you don't care to know output shape
    if kwargs == 'multi':
        features = extract_multi()
    elif kwargs == 'single':
        features = extract_single()
    # TODO: extract_1d
    extension = str(args.ext)
    compressed = args.compressed
    out_path = str(args.out_path)
    # TODO: figure out compression for file types other than txt/csv

    if extension == "hdf5":
        # (Recommended, default) save --> .hdf
        f = h5py.File("" + out_path + ".hdf5", "w")
        f.create_dataset(name=str(args.out_path), data=features)
    elif extension == "npy":  # god please don't actually do this
        outfile = "" + out_path
        np.save(file=outfile, allow_pickle=True, arr=features)
    else:
        if compressed:
            extension += ".gz"
        outfile = "" + out_path + "." + extension
        np.savetxt(fname=outfile, X=features, fmt='%1.6f')
    # TODO: (distant future) npz for the optional list of concat. 1d arrays

args = collect_args()
save_features()
gc.collect()
exit(0)
