# Convert .nii.gz to .npz

import os
import gc
import time
import numpy as np
import nibabel as nib

# this file path
dirname = os.path.dirname(__file__)

# constants
dataFolder = 'data'
outFolder = 'npz'
imgCnt = 210

dataPath = os.path.join(dirname, dataFolder)
outPath = os.path.join(dirname, outFolder)

# create output directory
os.makedirs(outPath, exist_ok=True)

# output dictionaries
x = {}    # images
ykid = {} # masks kidney
ytum = {} # masks tumor

# load cases
cases = os.listdir(dataPath) # 209 scans which are anotated
cases.sort()
cases = cases[:imgCnt]

for case in cases:
    imagePath = os.path.join(dataPath, case, 'imaging.nii.gz')
    masksPath = os.path.join(dataPath, case, 'segmentation.nii.gz')

    # load img with stopwatch for info
    stopwatch = time.time()

    imgarr = nib.load(imagePath).get_data().astype(np.uint8)
    mskarr = nib.load(masksPath).get_data().astype(np.uint8)

    print(f'case {case} was loaded in {time.time() - stopwatch} seconds')
    print(f'* img has minimum value: {np.min(imgarr)} and maximum value: {np.max(imgarr)}')
    print(f'* msk has minimum value: {np.min(mskarr)} and maximum value: {np.max(mskarr)}')

    # separate masks
    stopwatch = time.time()

    mask1arr = mskarr           # Reference
    mask2arr = np.copy(mskarr)  # Deep copy

    mask1arr[mask1arr != 1] = 0
    mask1arr[mask1arr == 1] = 1

    mask2arr[mask2arr != 2] = 0
    mask2arr[mask2arr == 2] = 1
    
    print(f'* case {case} masks were separated in {time.time() - stopwatch} seconds')
    print(f'* * mask1arr has minimum value: {np.min(mask1arr)} and maximum value: {np.max(mask1arr)}')
    print(f'* * mask2arr has minimum value: {np.min(mask2arr)} and maximum value: {np.max(mask2arr)}')

    for n in range(imgarr.shape[0]):
        img = imgarr[n,:,:] 
        kid = mask1arr[n,:,:]
        tum = mask2arr[n,:,:]
        
        if kid.shape[0] != 512 or kid.shape[1] != 512 or tum.shape[0] != 512 or tum.shape[1] != 512 or img.shape[0] != 512 or img.shape[1] != 512: 
            # image has unexpected shape, skip it (case 160 with shape 512x796)
            continue

        img = np.multiply(img, 1.0/255.0, dtype=np.float32)

        # store in dictionary if there is kid or tum
        if np.max(kid) == 1 or np.max(tum) == 1:
            x[f'{case}_{n}'] = img.reshape(1, imgarr.shape[1], imgarr.shape[2]).astype(np.float32)
            ykid[f'{case}_{n}'] = kid.reshape(1, mask1arr.shape[1], mask1arr.shape[2]).astype(np.float32)
            ytum[f'{case}_{n}'] = tum.reshape(1, mask2arr.shape[1], mask2arr.shape[2]).astype(np.float32)
            pass
        pass
    pass

stopwatch = time.time()

print(f'Saving was started. This operation can take a while (cca 30 minutes), be patient please...')

xPath = os.path.join(outPath, 'x.npz')
ykidPath = os.path.join(outPath, 'ykid.npz')
ytumPath = os.path.join(outPath, 'ytum.npz')

np.savez_compressed(ykidPath, **ykid)
del ykid
gc.collect()

np.savez_compressed(ytumPath, **ytum)
del ytum
gc.collect()

np.savez_compressed(xPath, **x)

print(f'Array with {len(x)} images was saved in {time.time() - stopwatch} seconds')
print(f'Done!')