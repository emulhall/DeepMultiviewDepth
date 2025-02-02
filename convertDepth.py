# File format for images.txt
# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)

# Example line
# 198 0.972561 0.0136273 0.190813 0.132398 -1.7855 -0.605043 0.0243369 1 frame_000197.png

import sys
import os
import numpy as np
from PIL import Image
from read_write_dense import read_array

def convertDepth(inDir, outDir):
    if not os.path.exists(inDir):
        print("Could not find directory:", inDir)
        quit()
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    files = os.listdir(inDir)
    for f in files:
        out = read_array(os.path.join(inDir, f))
        min_depth, max_depth = np.percentile(out, [5, 95])
        out[out < min_depth] = min_depth
        out[out > max_depth] = max_depth
        outIm = Image.fromarray(out).convert("L")

        outFile = f.split(".")[0] + ".pgm"
        outIm.save(os.path.join(outDir, outFile))


def convertSingleDepth(inFile):
    if not os.path.exists(inFile):
        print("Could not find file:", inFile)
        quit()

    out = read_array(inFile)
    min_depth, max_depth = np.percentile(out, [5, 95])
    out[out < min_depth] = min_depth
    out[out > max_depth] = max_depth
    outIm = Image.fromarray(out).convert("L")

    return outIm

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(sys.argv)
        print("Usage: convertDepth.py <input directory> <output directory>")
        quit()
    
    convertDepth(sys.argv[1], sys.argv[2])
