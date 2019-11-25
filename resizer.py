## Bulk image resizer

# This script simply resizes all the images in a folder to one-eigth their
# original size. It's useful for shrinking large cell phone pictures down
# to a size that's more manageable for model training.

# Usage: place this script in a folder of images you want to shrink,
# and then run it.

import numpy as np
import cv2
import os
import sys
if sys.argv[1] is not None:
	dir_path = sys.argv[1]
else:
	dir_path = os.getcwd()

print("Current Path: ",dir_path)

for filename in os.listdir(dir_path):
    # If the images are not .JPG images, change the line below to match the image type.
    if filename.endswith(".JPG") or filename.endswith(".jpg"):
        filename = os.path.join(dir_path,filename)	
        fileSize = os.path.getsize(filename)
        print("Image File: ",filename,fileSize)
        # file> 200KB to convert
        if fileSize > 200*1024:
	        image = cv2.imread(filename)
	        resized = cv2.resize(image,None,fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
	        cv2.imwrite(filename,resized)
