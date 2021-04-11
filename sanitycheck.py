# Export .nii.gz to .png for sanity check

import os
import time
import numpy as np
import nibabel as nib

from PIL import Image

# this file path
dirname = os.path.dirname(__file__)

# constants
dataFolder = 'data'
outFolder = 'png'

startCase = 0
caseCnt = 210
endCase = startCase + caseCnt

dataPath = os.path.join(dirname, dataFolder)
outPath = os.path.join(dirname, outFolder)

# create output directory
os.makedirs(outPath, exist_ok=True)

# load cases
cases = os.listdir(dataPath)
cases.sort()
cases = cases[:]

for case in cases[startCase:endCase]:
    imagePath = os.path.join(dataPath, case, 'imaging.nii.gz')
    masksPath = os.path.join(dataPath, case, 'segmentation.nii.gz')

    # load img with stopwatch for info
    stopwatch = time.time()

    imgarr = nib.load(imagePath).get_data().astype(np.uint8)
    mskarr = nib.load(masksPath).get_data().astype(np.uint8)

    print(f'case {case} was loaded in {time.time() - stopwatch} seconds')
    print(f'img has minimum value: {np.min(imgarr)} and maximum value: {np.max(imgarr)}')
    print(f'msk has minimum value: {np.min(mskarr)} and maximum value: {np.max(mskarr)}')

    # separate masks
    stopwatch = time.time()

    maskKidArr = mskarr             # Reference
    maskTumArr = np.copy(mskarr)    # Deep copy

    maskKidArr[maskKidArr != 1] = 0
    maskKidArr[maskKidArr == 1] = 255

    maskTumArr[maskTumArr != 2] = 0
    maskTumArr[maskTumArr == 2] = 255
    
    print(f'case {case} masks were separated in {time.time() - stopwatch} seconds')
    print(f'mask1arr has minimum value: {np.min(maskKidArr)} and maximum value: {np.max(maskKidArr)}')
    print(f'mask2arr has minimum value: {np.min(maskTumArr)} and maximum value: {np.max(maskTumArr)}')

    stopwatch = time.time()
    for n in range(imgarr.shape[0]):
        # image convert
        image = Image.fromarray(imgarr[n,:,:])
        kidmask = Image.fromarray(maskKidArr[n,:,:])
        tummask = Image.fromarray(maskTumArr[n,:,:])
        
        # image save
        image.save(os.path.join(outPath, f"{case}_{n}.png"))             # raw image
        kidmask.save(os.path.join(outPath, f"{case}_{n}_mask_kid.png"))  # kidney
        tummask.save(os.path.join(outPath, f"{case}_{n}_mask_tum.png"))  # tumor
        pass
    print(f'case {case} with {imgarr.shape[0]} images were saved in {time.time() - stopwatch} seconds')
    pass

print('Done!')