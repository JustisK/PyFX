import os, sys, re
import numpy as np
import pandas as pd
import skimage.io
import skimage.transform
from sklearn.preprocessing import LabelBinarizer
from keras.applications import InceptionV3
from keras.applications import imagenet_utils
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD, Adam

# Load dataset (any image-based dataset)
matches = [(re.match(r'^(([a-zA-Z]+)\d+\.png)', fname), path) for path, dirs, files in os.walk('./path/to/patches') for fname in files]
patches = [skimage.transform.resize( # resize image to (256, 256) TODO argv param based on patch size
        skimage.io.imread(os.path.join(path, match.group(1))), # open each image
    (256, 256)) for match, path in matches if match]
patches = imagenet_utils.preprocess_input(np.array(patches)) # change color channel order and shift mean color for ImageNet
labels = [match.group(2) for match, path in matches if match]
labels = LabelBinarizer().fit_transform(list(labels)) # one-hot encoding
print('patches', patches[0].shape, len(patches), 'labels', len(labels)) # TODO figure out if labels are needed at all (shouldn't be)

"""
# Construct model (using ImageNet weights)
inceptionV3 = InceptionV3(weights = "imagenet", include_top = False, input_shape = patches[0].shape)
# for layer in inceptionV3.layers:
#     layer.trainable = False
x = inceptionV3.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
"""

# TODO feature extraction FROM LAYER BEHIND SOFTMAX CLASSIFIER

features = inceptionV3.predict(x) # no idea if this will actually work
print(features[:25])

# END FEATURE EXTRACTION
predictions = Dense(21, activation="softmax")(x)
model = Model(inputs = [inceptionV3.input], outputs = [predictions])
model.compile(loss = "categorical_crossentropy", optimizer = Adam(lr=0.0001), metrics=["accuracy"])
model.summary()

# Train model
model.fit(x=patches, y=labels, batch_size=16, epochs = 10, validation_split=0.1, shuffle=True)

