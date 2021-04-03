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

def CustomUNet(INPUTHEIGHT=512, INPUTWIDTH=512, INPUTCHANNELS=1, OUTPUTCHANNELS=1):
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

    return model