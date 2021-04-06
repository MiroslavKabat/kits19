# loss functions used for training

from keras import backend as K

smooth = 1.0 # Used to prevent denominator 0

# IOU = Jaccard
def IOU(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def IOU_loss(y_true, y_pred):
    return 1.0 - IOU(y_true, y_pred)

# DICE = similar to Jaccard(IOU)
def DICE(y_true, y_pred):
    y_true_f = K.flatten(y_true) # y_true stretch to one dimension
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

def DICE_loss(y_true, y_pred):
    return 1.0 - DICE(y_true, y_pred)

# DICE + IOU
def DICE_IOU(y_true, y_pred):
    return (DICE(y_true, y_pred) + IOU(y_true, y_pred)) / 2.0

def DICE_IOU_loss(y_true, y_pred):
    return 1.0 - DICE_IOU(y_true, y_pred)