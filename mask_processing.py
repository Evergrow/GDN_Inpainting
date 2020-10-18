import cv2
import os
import random
import numpy as np

src_path = ''       # raw irregular mask path
tar_path = ''       # processing mask path

# Read image
dirs_list = os.listdir(src_path)

list10 = []
list20 = []
list30 = []
list40 = []
list50 = []
list60 = []

num = 0
sum = 0.0
thr = 30000
margin = True/False   # we set margin in range [0-10], [10-20]


for dir in dirs_list:
    # convert to binary mask
    mask = cv2.imread(src_path + dir)
    mask = mask[:, :, 0]
    h, w = mask.shape
    bi = mask / 255
    
    # calculate hole ratio
    bg = np.sum(bi)
    hr = bg / (h * w)
    
    # rotate mask in random
    rand = random.randint(-180, 180)
    M = cv2.getRotationMatrix2D((mask.shape[1] / 2, mask.shape[0] / 2), rand, 1.5)
    mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
    
    # erode mask in random
    low = int(15 - hr) if hr < 14 else 1
    high = int(28 - hr) if hr < 28 else 1
    rand = random.randint(low, high)
    kernel = np.ones((rand, rand), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    # crop mask in random
    x = random.randint(0, 256)
    y = random.randint(0, 256)
    mask = mask[y:y + 256, x:x + 256]
    
    # invert mask value
    mask = 255 - mask
    mask = (mask > 127.5).astype(np.uint8) * 255
    
    # add margin
    if margin:
        mask = (mask > 127.5).astype(np.uint8) * 255
        mask = mask[23:233,23:233]
        output = np.zeros([256, 256])
        output[23:233,23:233] = mask
        mask = output
    
    # new hole ratio
    h, w = mask.shape
    bg = np.sum(mask) / 256
    hr = bg / (h * w)
    
    # set name
    s = str(num)
    while len(s) < 6:
        s = '0' + s
    
    # save
    if hr < 0.1:
        list10.append(dir)
        if len(list10) < thr:
            cv2.imwrite(tar_path + '/mask0-10/' + s + '.png', mask)
        else:
            continue
    elif hr < 0.2:
        list20.append(dir)
        if len(list20) < thr:
            cv2.imwrite(tar_path + '/mask10-20/' + s + '.png', mask)
        else:
            continue
    elif hr < 0.3:
        list30.append(dir)
        if len(list30) < thr:
            cv2.imwrite(tar_path + '/mask20-30/' + s + '.png', mask)
        else:
            continue
    elif hr < 0.4:
        list40.append(dir)
        if len(list40) < thr:
            cv2.imwrite(tar_path + '/mask30-40/' + s + '.png', mask)
        else:
            continue
    elif hr < 0.5:
        list50.append(dir)
        if len(list50) < thr:
            cv2.imwrite(tar_path + '/mask40-50/' + s + '.png', mask)
        else:
            continue
    elif hr < 0.6:
        list60.append(dir)
        if len(list60) < thr:
            cv2.imwrite(tar_path + '/mask50-60/' + s + '.png', mask)
        else:
            continue
    else:
        continue
    
    num += 1
    sum += hr

print(len(list10), len(list20), len(list30), len(list40), len(list50), len(list60))
print(sum / len(dirs_list))
