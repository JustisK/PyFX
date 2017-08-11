"""
PyFX v0.1
-------------------
MIT License

Copyright (c) 2017 Keegan T. O. Justis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import re
import gc
import gzip, shutil
import argparse

import numpy as np
import h5py

import skimage.io
import skimage.transform
from sklearn.feature_extraction.image import extract_patches_2d

from keras.applications import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten
# from keras import backend as K  #TODO: actually use K.reshape


def collect_args():
    """
    Collects command line arguments from invocation.
    :return: argparse object containing parsed command line arguments
    """
    # Command line arguments: image in-path, feature out-path, ext for output
    parser = argparse.ArgumentParser(description="""Perform InceptionV3
     feature extraction on images.""")

    # TODO: nargs, document these - explain each type
    # TODO: case-insensitivity changes
    parser.add_argument(nargs='?', type=str, dest='extractor',
                        default='multi', action='store')
    # TODO: -silent (no prompting) w/ default=prompt for args
    parser.add_argument(nargs=1, type=bool, dest='silent',
                        default=True, action='store')
    parser.add_argument(nargs='?', type=str, dest='img_path',
                        default='./images', action='store')
    """
    # Need to figure out how to integrate this with regex.
    parser.add_argument(nargs='?', type=str, dest='img_type',
                        default='png', action='store')
    """
    parser.add_argument(nargs='?', type=str, dest='out_path',
                        default='./output/features', action='store')
    parser.add_argument(nargs='?', type=str, dest='ext',
                        default='hdf5', action='store')
    """
    TODO: figure out why this and other boolean args get set True
    when defaults are False and False is passed to them in xterm.
    """
    parser.add_argument(nargs='?', type=bool, dest='compressed',
                        default=False, action='store')
    parser.add_argument(nargs='?', type=bool, dest='flatten',
                        default=False, action='store')
    argv = parser.parse_args()

    compressed = argv.compressed
    extension = argv.ext

    if extension != ("csv" or "txt"):
        # TODO: string formatting here is bad
        print("""WARNING: non-text output (bin, npy, hdf5) is incompressible for now.
        \nOutput will not be compressed.""")
    elif not compressed:
        print("""WARNING: non-compressed csv output is extremely large.
        Recommend re-running with compressed=True.""")
    return argv


def extract_multi():
    """
    extract_multi

    Extracts feature data for each member in a directory containing .png images

    :return: Keras tensor containing extracted features.
    """

    # Load dataset (any image-based dataset)
    # TODO: argv for file types other than png
    matches = [(re.match(r'^(([a-zA-Z]+)\d+\.png)', fname), path)
               for path, dirs, files in os.walk('' + str(args.img_path))
               for fname in files]
    # Resize / regularize image 'patches'
    patches = [skimage.transform.resize(  # resize image to (256, 256)
        skimage.io.imread(os.path.join(path, match.group(1))),  # open each img
        (256, 256)) for match, path in matches if match]

    # Pre-process for InceptionV3
    patches = preprocess_input(np.array(patches))

    # Construct model (using ImageNet weights)
    inception = InceptionV3(weights="imagenet", include_top=False,
                            input_shape=patches[0].shape)

    # Isolate pre-softmax outputs
    x = inception.output

    # Flatten to 1d
    if args.flatten or args.ext == 'csv':
        x = Flatten()(x)
        # TODO: K.reshape(x) to 2d for csv

    # Construct extractor model
    extractor = Model(inputs=[inception.input], outputs=[x])

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

    # Load target image
    target = skimage.io.imread(args.img_path).astype('float32')
    # TODO: patch overlap
    patches = extract_patches_2d(target, (256, 256))

    """
    # Regularize to 256x256
    # TODO: allow different patch/resize dimensions parametrically

    for patch in patches:
        skimage.transform.resize(patch, (256, 256))

    """

    # Pre-process for InceptionV3
    patches = preprocess_input(np.array(patches))

    # Construct model (using ImageNet weights)
    inception = InceptionV3(weights="imagenet", include_top=False,
                            input_shape=patches[0].shape)

    # Isolate pre-softmax outputs
    x = inception.output

    # Experimental - flatten to 1d
    if args.flatten or args.ext == 'csv':
        x = Flatten()(x)

    # Construct extractor model
    extractor = Model(inputs=[inception.input], outputs=[x])

    # Extract features with Model.predict()
    features = extractor.predict(x=patches, batch_size=2)
    # TODO (distant future): K.reshape(x) to 2d
    # features = K.reshape(features, (36, 2048))
    # TODO (distant future): get rid of zero-padding

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
    return target


def save_features():
    """
    Writes extracted feature vectors into a binary or text file, per args.
    :return: none
    """

    extractor = args.extractor
    features = []

    if extractor == 'multi':
        features = extract_multi()
    elif extractor == 'single':
        features = extract_single()
    # TODO: extract_1d

    print("Output shape: ", features.shape)  # comment out if you don't care to know output shape

    extension = str(args.ext)
    compress = args.compressed
    out_path = str(args.out_path)

    # TODO: figure out compression for file types other than txt/csv
    outfile = "" + out_path
    out_full = outfile + "." + extension
    if extension == "hdf5":
        # (Recommended, default) save to .hdf5
        f = h5py.File("" + out_path + ".hdf5", "w")
        f.create_dataset(name=str(args.out_path), data=features)
        if compress:
            with open(out_full) as f_in:
                outfile_gz = out_full + ".gz"
                with gzip.open(outfile_gz, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    elif extension == "npy":  # god please don't actually do this
        # Save to .npy binary (numpy) - incompressible (as of now)
        np.save(file=outfile, allow_pickle=True, arr=features)
        if compress:
            with open(out_full) as f_in:
                outfile_gz = out_full + ".gz"
                with gzip.open(outfile_gz, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    elif extension == "csv":
        # Save to .csv (or, .csv.gz if args.compressed==True)
        # This option is natively compressible.
        if compress:
            extension += ".gz"
        outfile = "" + out_path + "." + extension
        np.savetxt(fname=outfile, X=features, fmt='%1.5f')
    # TODO: (distant future) npz for the optional list of concat. 1d arrays


def main():
    """
    Execute feature extraction.
    :return: None. Should exit with code 0 on success.
    """
    save_features()
    gc.collect()
    exit(0)  # TODO: check - change exit code for failure


def fill_args():
    prompt = ""
    return prompt


# PROMPTS = {}  # TODO: put prompts here
args = collect_args()  # TODO: get rid of global variables

# TODO: add extractor option that passes out features in a string

main()
