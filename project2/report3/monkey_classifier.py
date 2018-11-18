# -*- coding: utf-8 -*-
"""Monkey Classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1P5b5s2b2HlABgEa3JQ2nju_3Z1RenuWE

# Monkey Species Identifier

## Initialization

### Imports
"""

import numpy as np
import joblib

from google.colab import drive

from sklearn.metrics import precision_recall_fscore_support

# Mount google drive
drive.mount("drive")

"""### Load Dataset"""

DATASET_PATH = "drive/My Drive/dataset/dataset_monkey.joblib"
X_train, Y_train = joblib.load(DATASET_PATH)

DATASET_PATH = "drive/My Drive/dataset/dataset_monkey_test.joblib"
X_test, Y_test = joblib.load(DATASET_PATH)

"""### Set variables"""

HEIGHT = X_train.shape[1]
WIDTH = X_train.shape[2]
CHANNELS = X_train.shape[3]

TRAIN_SAMPLES = X_train.shape[0]
TEST_SAMPLES = X_test.shape[0]
BATCH_SIZE = 32
CLASSES = 10
EPOCH_STEPS = int(TRAIN_SAMPLES / BATCH_SIZE)
EPOCHS = 50

NN_OUTPUT_HEIGHT = 7
NN_OUTPUT_WIDTH = 7
NN_OUTPUT_CHANNELS = 2048

"""## Feature Extraction - Xception Network"""

from keras.applications.xception import Xception

nn_model = Xception(
  weights="imagenet",
  input_shape=(HEIGHT, WIDTH, CHANNELS),
  include_top=False
)

"""### Training dataset"""

X_train_reduced = np.zeros(
    (TRAIN_SAMPLES, NN_OUTPUT_HEIGHT, NN_OUTPUT_WIDTH,
        NN_OUTPUT_CHANNELS), dtype=np.float32)

for i in range(TRAIN_SAMPLES):
    X_train_reduced[i, :, :, :] = nn_model.predict(
        X_train[i:i+1, :, :, :])

# Cleanup memory
del X_train
  
X_train = np.reshape(X_train_reduced, (TRAIN_SAMPLES, 7 * 7 * 2048))

"""### Test dataset"""

X_test_reduced = np.zeros(
    (TEST_SAMPLES, NN_OUTPUT_HEIGHT, NN_OUTPUT_WIDTH,
        NN_OUTPUT_CHANNELS), dtype=np.float32)

for i in range(TEST_SAMPLES):
    X_test_reduced[i, :, :, :] = nn_model.predict(X_test[i:i+1, :, :, :])

# Cleanup memory
del X_test

X_test = np.reshape(X_test_reduced, (TEST_SAMPLES, 7 * 7 * 2048))

"""## Feature Extraction -  PCA"""

from sklearn.decomposition import PCA

pca = PCA(n_components=3291)

"""### Training dataset"""

X_train = pca.fit_transform(
    X_train.reshape(TRAIN_SAMPLES, HEIGHT*WIDTH*CHANNELS))

"""### Test dataset"""

X_test = pca.transform(
    X_test.reshape(TEST_SAMPLES, HEIGHT*WIDTH*CHANNELS))

"""# Classification - Neural Network

### Model
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=7*7*2048))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

"""### Training"""

history = model.fit(
  X_train, 
  Y_train, 
  epochs=25,
  batch_size=BATCH_SIZE,
  shuffle=True
)

# Predict values
Y_pred = model.predict(X_test)

# Convert predicted values to labels
Y_pred_labels = [x.argmax() for x in Y_pred]
Y_test_labels = [x.argmax() for x in Y_test]

print(precision_recall_fscore_support(Y_test_labels, Y_pred_labels))

"""# Classification - Support Vector Machine

### Model
"""

from sklearn.svm import SVC

model = SVC(C=10.0)

# Convert on hot encoding to labels
Y_train_labels = [x.argmax() for x in Y_train]

"""### Training"""

model.fit(X_train, Y_train_labels)

# Predict values
Y_pred_labels = model.predict(X_test)

# Convert test one hot encoding to labels
Y_test_labels = [x.argmax() for x in Y_test]

print(precision_recall_fscore_support(Y_test_labels, Y_pred_labels))