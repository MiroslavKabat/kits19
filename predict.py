# Prediction with last model
# TODO: you can merge heatmap of prediction and with mask to better comparision

import os
import gc
import time

import numpy as np
import keras

from keras.models import load_model
from keras import backend as K

from datetime import datetime
from tqdm import tqdm

from PIL import Image

# this file path
dirname = os.path.dirname(__file__)
modelsFolder = 'models'
datasetFolder = 'npz'
outputFolder = 'predictions'

bestModelName = 'model_best.h5'

modelsDir = os.path.join(dirname, modelsFolder)
outputDir = os.path.join(dirname, outputFolder)
pathToImages = os.path.join(dirname, datasetFolder, "x.npz")          # Images
pathToMasks = os.path.join(dirname, datasetFolder, "ykid.npz")        # kidney .. ykid.npz | tumor .. ytum.npz 

inputHeight = 512
inputWidth = 512

countOfImagesFromDataset = 10
startIndexFromDataset = 10000
endIndexFromDataset = startIndexFromDataset + countOfImagesFromDataset

fake3channels = True # True if your model expect ?:?:3

# load model
modelPaths = os.listdir(modelsDir)
modelPaths.sort()

for modelPath in modelPaths[-1:]:

    bestModelPath = os.path.join(modelsDir, modelPath, bestModelName)
    if not os.path.exists(bestModelPath):
        continue

    # load model
    model = load_model(bestModelPath, compile=False)

    # load data
    images = np.load(pathToImages, None, True)
    masks = np.load(pathToMasks, None, True)
    keys = images.files
    keys = keys[startIndexFromDataset:endIndexFromDataset] # take all -> files[:] or take 10 for example -> files[:10]

    try:
        # predict
        for key in keys:
            X = images[key]
            if fake3channels:
                X = X.reshape(1, inputHeight, inputWidth, 1)
                X = np.concatenate((X, X, X), axis=3)
                pass
            
            prediction = model.predict(X)
            prediction = prediction * 255.0
            prediction = prediction.astype(np.uint8).reshape(512, 512)

            image = Image.fromarray(prediction)
            os.makedirs(os.path.join(modelsDir, modelPath , outputFolder), exist_ok=True)           
            image.save(os.path.join(modelsDir, modelPath , outputFolder, f'{key}_prediction.png'))
            pass
        pass
    except:
        print(f'Model fall to exception: {modelPath}')
        continue
    pass

print('Done!')