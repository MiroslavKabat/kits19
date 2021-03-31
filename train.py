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
from keras.layers import Input, Cropping2D, UpSampling2D, Concatenate, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam, Adadelta
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

from datetime import datetime
from tqdm import tqdm

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
COUNTOFIMAGESFROMDATASET = 16220            # you can use large number for all images like 999999
CHERRYPICKING = True                        # Pick only valid images from dataset
CHERRYMIN = 0.01                            # used only if CHERRYPICKING is True
CHERRYMAX = 1.00                            # used only if CHERRYPICKING is True

# optimizer - Hyperparameter
LEARNINGRATE = 0.01
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
keys = keys[:COUNTOFIMAGESFROMDATASET] # take all -> files[:] or take 10 for example -> files[:10]

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
PADDING = "same" # "valid"
ACTIVATION = "relu"
INITIALIZER = "he_normal"
KERNELSIZE = (3,3)
POOLSIZE = (2,2)
UPSIZE = (2,2)
STRIDES = (1,1)

## input
p0 = Input(shape=(INPUTHEIGHT, INPUTWIDTH, INPUTCHANNELS), name="input")

## compressing down
c1 = Conv2D(64, KERNELSIZE, STRIDES, PADDING, activation=ACTIVATION, kernel_initializer=INITIALIZER)(p0)
c2 = Conv2D(64, KERNELSIZE, STRIDES, PADDING, activation=ACTIVATION, kernel_initializer=INITIALIZER)(c1)
p1 = MaxPooling2D(POOLSIZE, POOLSIZE, PADDING)(c2)

c3 = Conv2D(128, KERNELSIZE, STRIDES, PADDING, activation=ACTIVATION, kernel_initializer=INITIALIZER)(p1)
c4 = Conv2D(128, KERNELSIZE, STRIDES, PADDING, activation=ACTIVATION, kernel_initializer=INITIALIZER)(c3)
p2 = MaxPooling2D(POOLSIZE, POOLSIZE, PADDING)(c4)

c5 = Conv2D(256, KERNELSIZE, STRIDES, PADDING, activation=ACTIVATION, kernel_initializer=INITIALIZER)(p2)
c6 = Conv2D(256, KERNELSIZE, STRIDES, PADDING, activation=ACTIVATION, kernel_initializer=INITIALIZER)(c5)
p3 = MaxPooling2D(POOLSIZE, POOLSIZE, PADDING)(c6)

c7 = Conv2D(512, KERNELSIZE, STRIDES, PADDING, activation=ACTIVATION, kernel_initializer=INITIALIZER)(p3)
c8 = Conv2D(512, KERNELSIZE, STRIDES, PADDING, activation=ACTIVATION, kernel_initializer=INITIALIZER)(c7)
p4 = MaxPooling2D(POOLSIZE, POOLSIZE, PADDING)(c8)

## bottleneck
c9 = Conv2D(1024, KERNELSIZE, STRIDES, PADDING, activation=ACTIVATION, kernel_initializer=INITIALIZER)(p4)
c10 = Conv2D(1024, KERNELSIZE, STRIDES, PADDING, activation=ACTIVATION, kernel_initializer=INITIALIZER)(c9)

## expanding up
up1 = UpSampling2D(UPSIZE)(c10)
crop1 = Cropping2D((0,0), None)(c8)
conc1 = concatenate([up1, crop1], axis=3)
c11 = Conv2D(512, KERNELSIZE, STRIDES, PADDING, activation=ACTIVATION, kernel_initializer=INITIALIZER)(conc1)
c12 = Conv2D(512, KERNELSIZE, STRIDES, PADDING, activation=ACTIVATION, kernel_initializer=INITIALIZER)(c11)

up2 = UpSampling2D(UPSIZE)(c12)
crop2 = Cropping2D((0,0), None)(c6)
conc2 = concatenate([up2, crop2], axis=3)
c13 = Conv2D(256, KERNELSIZE, STRIDES, PADDING, activation=ACTIVATION, kernel_initializer=INITIALIZER)(conc2)
c14 = Conv2D(256, KERNELSIZE, STRIDES, PADDING, activation=ACTIVATION, kernel_initializer=INITIALIZER)(c13)

up3 = UpSampling2D(UPSIZE)(c14)
crop3 = Cropping2D((0,0), None)(c4)
conc3 = concatenate([up3, crop3], axis=3)
c15 = Conv2D(128, KERNELSIZE, STRIDES, PADDING, activation=ACTIVATION, kernel_initializer=INITIALIZER)(conc3)
c16 = Conv2D(128, KERNELSIZE, STRIDES, PADDING, activation=ACTIVATION, kernel_initializer=INITIALIZER)(c15)

up4 = UpSampling2D(UPSIZE)(c16)
crop4 = Cropping2D((0,0), None)(c2)
conc4 = concatenate([up4, crop4], axis=3)
c17 = Conv2D(64, KERNELSIZE, STRIDES, PADDING, activation=ACTIVATION, kernel_initializer=INITIALIZER)(conc4)
c18 = Conv2D(64, KERNELSIZE, STRIDES, PADDING, activation=ACTIVATION, kernel_initializer=INITIALIZER)(c17)

## output
c19 = Conv2D(OUTPUTCHANNELS, (1,1), (1,1), PADDING, activation="sigmoid")(c18)

model = Model(inputs=p0, outputs=c19, name="u-net")

model.summary()

# create optimizer
optimizer = keras.optimizers.Adadelta(learning_rate=LEARNINGRATE, rho=RHO, epsilon=EPSILON, name="Adadelta", decay=DECAY )

# define custom loss function
def IOU(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def IOU_loss(y_true, y_pred):
    return 1.0 - IOU(y_true, y_pred)

# compile model with loss function
# model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics="accuracy")
model.compile(optimizer=optimizer, loss=[IOU_loss], metrics=[IOU])

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
    validation_split=VALIDATIONSPLIT,
    validation_data=None,
    shuffle=True,
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