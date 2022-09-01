import os
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

ACTIVATION_FN = 'elu'
filters = 32

input_layer = Input(shape=(64000,1))
conv1 = Conv1D(filters, kernel_size=3, padding='same')(input_layer)
batch1 = BatchNormalization()(conv1)
act1 = Activation(ACTIVATION_FN)(batch1)
pool1 = MaxPool1D(strides=2)(act1)

## 32000, 128
conv2 = Conv1D(filters *2, kernel_size=3, padding='same')(pool1)
batch2 = BatchNormalization()(conv2)
act2 = Activation(ACTIVATION_FN)(batch2)
pool2 = MaxPool1D(strides=2)(act2)

## 16000, 256
conv3 = Conv1D(filters *4, kernel_size=3, padding='same')(pool2)
batch3 = BatchNormalization()(conv3)
act3 = Activation(ACTIVATION_FN)(batch3)
pool3 = MaxPool1D(strides=2)(act3)

## 8000, 512
conv4 = Conv1D(filters*8, kernel_size=3, padding='same')(pool3)
batch4 = BatchNormalization()(conv4)
act4 = Activation(ACTIVATION_FN)(batch4)
pool4 = MaxPool1D(strides=2)(act4)

## 4000, 1024
conv5 = Conv1D(filters *16, kernel_size=3, padding='same')(pool4)
batch5 = BatchNormalization()(conv5)
act5 = Activation(ACTIVATION_FN)(batch5)
pool5 = MaxPool1D(strides=2)(act5)

## 2000
conv6 = Conv1D(filters *16, kernel_size=3, padding='same')(pool5)
batch6 = BatchNormalization()(conv6)
flat1 = Flatten()(batch6)
dense1 = Dense(200, activation='relu')(flat1)
batch7 = BatchNormalization()(dense1)
dense2 = Dense(20, activation='relu')(batch7)
# dense2 = Dense(20, activation='relu')(dense1)
output_layer = Dense(1, activation='sigmoid')(dense2)

model = Model(input_layer, output_layer)

model.summary()

# model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
# history = model.fit(train, test,, batch_size=5, epochs=5, shuffle=True, verbose=2)
# model.predict()

# tf.cast()