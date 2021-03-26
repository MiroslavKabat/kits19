# Export .nii.gz to .png for sanity check

# prerequisities:
# downloaded get_imaging.py

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
imgcnt = 210

dataPath = os.path.join(dirname, dataFolder)
outPath = os.path.join(dirname, outFolder)

# create output directory
os.makedirs(outPath, exist_ok=True)

# load cases
cases = os.listdir(dataPath)
cases.sort()
cases = cases[:imgcnt]

for case in cases:
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

    mask1arr = mskarr             # Reference
    mask2arr = np.copy(mskarr)    # Deep copy

    mask1arr[mask1arr != 1] = 0
    mask1arr[mask1arr == 1] = 255

    mask2arr[mask2arr != 2] = 0
    mask2arr[mask2arr == 2] = 255
    
    print(f'case {case} masks were separated in {time.time() - stopwatch} seconds')
    print(f'mask1arr has minimum value: {np.min(mask1arr)} and maximum value: {np.max(mask1arr)}')
    print(f'mask2arr has minimum value: {np.min(mask2arr)} and maximum value: {np.max(mask2arr)}')

    stopwatch = time.time()
    for n in range(imgarr.shape[0]):
        # image convert
        image = Image.fromarray(imgarr[n,:,:])
        mask1 = Image.fromarray(mask1arr[n,:,:])
        mask2 = Image.fromarray(mask2arr[n,:,:])
        
        # image save
        image.save(os.path.join(outPath, f"{case}_{n}.png"))        # raw image
        mask1.save(os.path.join(outPath, f"{case}_{n}_mask1.png"))  # kidney
        mask2.save(os.path.join(outPath, f"{case}_{n}_mask2.png"))  # tumor
        pass
    print(f'case {case} with {imgarr.shape[0]} images was saved in {time.time() - stopwatch} seconds')
    pass

print('Done!')