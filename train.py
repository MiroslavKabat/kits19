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
from unetmodels import *

# setup PC
keras.backend.set_image_data_format('channels_last')

# stamp
now = datetime.now()
timestamp = str(now.year).zfill(4) + str(now.month).zfill(2) + str(now.day).zfill(2) + str(now.hour).zfill(2) + str(now.minute).zfill(2)

# constants & variables
DIRNAME = os.path.dirname(__file__)

DATASETNAME = "npz"
OUTPUTDIRECTORY = f"models/{timestamp}"

PATHTOFILENAMES = os.path.join(DIRNAME, DATASETNAME, "keys.txt")    # Images names
PATHTOIMAGES = os.path.join(DIRNAME, DATASETNAME, "x.npz")          # Images
PATHTOMASKS = os.path.join(DIRNAME, DATASETNAME, "ykid.npz")        # kidney .. ykid.npz | tumor .. ytum.npz 

MODELFILEPATH = os.path.join(DIRNAME, OUTPUTDIRECTORY, "model.h5")           # trained model is saved in to this file h5
ONNXMODELFILEPATH = os.path.join(DIRNAME, OUTPUTDIRECTORY, "model.onnx")     # trained model is saved in to this file onnx
OUTPUTLOGPATH = os.path.join(DIRNAME, OUTPUTDIRECTORY, "train.csv")
TENSORBOARDPATH = os.path.join(DIRNAME, OUTPUTDIRECTORY, "./logs")
CHECKPOINTPATH = os.path.join(DIRNAME, OUTPUTDIRECTORY, "model.{epoch:02d}-{val_loss:.5f}.h5")
BESTCHECKPOINTPATH = os.path.join(DIRNAME, OUTPUTDIRECTORY, "model_best.h5")

# data loading
startImage = 0
cntOfImagesFromDataset = 1000 # 16220            # you can use large number for all images like 999999
endImage = startImage + cntOfImagesFromDataset

CHERRYPICKING = True                             # Pick only valid images from dataset
CHERRYMIN = 0.01                                 # used only if CHERRYPICKING is True
CHERRYMAX = 1.00                                 # used only if CHERRYPICKING is True
FAKE3CHANNELS = True

# optimizer - Hyperparameter
LEARNINGRATE = 0.0001
RHO = 0.95
EPSILON = 1e-7
DECAY = 0

# U-NET architecture
INPUTCHANNELS = 1
INPUTHEIGHT = 512
INPUTWIDTH = 512

OUTPUTCHANNELS = 1

# training constants
CALLBACKPERIODCHECKPOINT = 10
BATCHSIZE = 3
EPOCHS = 100
VERBOSE = 1    # 0 .. silent | 1 .. every batch | 2 .. every epoch
VALIDATIONSPLIT = 0.2
VALIDATIONFREQUENCY = 1
MAXQUEUQSIZE = 10

# load data
stopwatch = time.time()
print(f'Loading data ...')

images = np.load(PATHTOIMAGES, None, True)
masks = np.load(PATHTOMASKS, None, True)
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
    if CHERRYPICKING:
        sum = float(yarr.sum())
        area = float(yarr.size)
        sumratio = sum / area

        if sumratio < CHERRYMIN or sumratio > CHERRYMAX:
            continue

    if FAKE3CHANNELS:
        xarr = xarr.reshape(1, INPUTHEIGHT, INPUTHEIGHT, 1)
        xarr = np.concatenate((xarr, xarr, xarr),axis=3)
        yarr = yarr.reshape(1, INPUTHEIGHT, INPUTWIDTH, 1)
        pass

    x.append(xarr)
    y.append(yarr)
    pass

# concatenate selected images and masks
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

# model = sm.Unet()
model = sm.Unet('resnet34')
# model = CustomUNet() # our custom UNet

model.summary() # info about model

# create optimizer
optimizer = keras.optimizers.Adadelta(learning_rate=LEARNINGRATE, rho=RHO, epsilon=EPSILON, name="Adadelta", decay=DECAY )
# optimizer = keras.optimizers.Adam(learning_rate=LEARNINGRATE)

# compile model with loss function
# model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics="accuracy")
model.compile(optimizer=optimizer, loss=[DICE_IOU_loss], metrics=[DICE_IOU])
# model.compile(optimizer=optimizer, loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])

# callbacks
callbacks = [
    callbacks.ModelCheckpoint(filepath=CHECKPOINTPATH, verbose=0, save_best_only=True, save_weights_only=False, period=CALLBACKPERIODCHECKPOINT),
    callbacks.ModelCheckpoint(filepath=BESTCHECKPOINTPATH, verbose=0, save_best_only=True, save_weights_only=False, period=1),
    callbacks.TensorBoard(log_dir=TENSORBOARDPATH, profile_batch=0),
    callbacks.CSVLogger(OUTPUTLOGPATH, separator=";", append=True)
]

print(f'Model builded in {time.time() - stopwatch} seconds')

# create output dir
os.makedirs(OUTPUTDIRECTORY, exist_ok=True)

# start training 
model.fit(
    x=X,
    y=Y,
    batch_size=BATCHSIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    callbacks=callbacks,
    validation_split=0.2, # VALIDATIONSPLIT,
    validation_data=None,  # (X, Y)
    shuffle=False,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=VALIDATIONFREQUENCY,
    max_queue_size=MAXQUEUQSIZE,
    workers=1,
    use_multiprocessing=False,
)

# save trained model .h5
model.save(filepath=MODELFILEPATH)

# # convert to onnx model (only used for deploy / you can use h5 to valid model)
# onnx_model = keras2onnx.convert_keras(model, "", "", int(7))
# onnx.save_model(onnx_model, ONNXMODELFILEPATH)

print("Done!")