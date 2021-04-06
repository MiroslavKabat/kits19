# Train model

import os
import gc
import time

import numpy as np
import keras
import onnx
import keras2onnx

from keras import callbacks
from keras.optimizers import Optimizer
from keras.models import Model
from keras.optimizers import Adam, Adadelta
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

import segmentation_models as sm

from datetime import datetime
from tqdm import tqdm

from lossfunctions import *

# setup PC
keras.backend.set_image_data_format('channels_last')

# stamp
now = datetime.now()
timestamp = str(now.year).zfill(4) + str(now.month).zfill(2) + str(now.day).zfill(2) + str(now.hour).zfill(2) + str(now.minute).zfill(2)

# constants & variables
dirname = os.path.dirname(__file__)

datasetDir = "npz"
outputDir = f"models/{timestamp}"

pathToImages = os.path.join(dirname, datasetDir, "x.npz")          # Images
pathToMasks = os.path.join(dirname, datasetDir, "ykid.npz")        # kidney .. ykid.npz | tumor .. ytum.npz 

modelFilePath = os.path.join(dirname, outputDir, "model.h5")           # trained model is saved in to this file h5
outputLogPath = os.path.join(dirname, outputDir, "train.csv")
tensorboardPath = os.path.join(dirname, outputDir, "./logs")
checkpointPath = os.path.join(dirname, outputDir, "model.{epoch:02d}-{val_loss:.5f}.h5")
bestCheckpointPath = os.path.join(dirname, outputDir, "model_best.h5")

# data loading
startImage = 0
cntOfImagesFromDataset = 10000 # 16220           # you can use large number for all images like 999999
endImage = startImage + cntOfImagesFromDataset

cherryPicking = True                             # Pick only valid images from dataset
cherryMin = 0.01                                 # used only if cherryPicking is True
cherryMax = 1.00                                 # used only if cherryPicking is True
fake3channels = True

# optimizer - Hyperparameter
learningRate = 0.001
rho = 0.95
epsilon = 1e-7
decay = 0

# U-NET architecture
architecture = 'resnet34'

inputHeight = 512
inputWidth = 512

# training constants
callbackPeriodCheckpoint = 10
batchsize = 4
epochs = 100
verbose = 2    # 0 .. silent | 1 .. every batch | 2 .. every epoch
validationSplit = 0.2
validationPeriod = 1
maxQueueSize = 10

# load data
stopwatch = time.time()
print(f'Loading data ...')

images = np.load(pathToImages, None, True)
masks = np.load(pathToMasks, None, True)
keys = images.files
keys = keys[startImage:endImage] # take all -> files[:] or take 10 for example -> files[:10]

# select data in to single array
x = [] # images
y = [] # masks
idx = 0
for file in tqdm(keys): 
    xarr = np.array(images[file])
    yarr = np.array(masks[file])

    # cherry picking
    if cherryPicking:
        sum = float(yarr.sum())
        area = float(yarr.size)
        sumratio = sum / area

        if sumratio < cherryMin or sumratio > cherryMax:
            continue

    if fake3channels:
        xarr = xarr.reshape((1, inputHeight, inputHeight, 1))
        xarr = np.concatenate((xarr, xarr, xarr),axis=3)
        yarr = yarr.reshape((1, inputHeight, inputWidth, 1))
        pass

    x.append(xarr)
    y.append(yarr)
    pass

# concatenate selected images and masks
print(f"Concatenation ...")
X = np.concatenate(x)
Y = np.concatenate(y)

print(f"Volume: {X.shape[0]}")

# force clean up memory
del x
del y
del images
del masks
gc.collect()

print(f'Data loaded in {time.time() - stopwatch} seconds')

# build model
stopwatch = time.time()
print(f'Building model ..')

model = sm.Unet(architecture)

model.summary() # info about model

# create optimizer
optimizer = keras.optimizers.Adadelta(learning_rate=learningRate, rho=rho, epsilon=epsilon, name="Adadelta", decay=decay )
# optimizer = keras.optimizers.Adam(learning_rate=learningRate)

# compile model with loss function
# model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics="accuracy")
model.compile(optimizer=optimizer, loss=[DICE_IOU_loss], metrics=[DICE_IOU])
# model.compile(optimizer=optimizer, loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])

# callbacks
callbacks = [
    callbacks.ModelCheckpoint(filepath=checkpointPath, verbose=0, save_best_only=True, save_weights_only=False, period=callbackPeriodCheckpoint),
    callbacks.ModelCheckpoint(filepath=bestCheckpointPath, verbose=0, save_best_only=True, save_weights_only=False, period=1),
    callbacks.TensorBoard(log_dir=tensorboardPath, profile_batch=0),
    callbacks.CSVLogger(outputLogPath, separator=";", append=True)
]

print(f'Model builded in {time.time() - stopwatch} seconds')

# create output dir
os.makedirs(outputDir, exist_ok=True)

# start training 
model.fit(
    x=X,
    y=Y,
    batch_size=batchsize,
    epochs=epochs,
    verbose=verbose,
    callbacks=callbacks,
    validation_split=0.2,  # validationSplit,
    validation_data=None,  # (X, Y)
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=validationPeriod,
    max_queue_size=maxQueueSize,
    workers=1,
    use_multiprocessing=False,
)

# save trained model .h5
model.save(filepath=modelFilePath)

print("Done!")