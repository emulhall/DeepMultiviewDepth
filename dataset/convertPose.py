# File format for images.txt
# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)

# Example line
# 198 0.972561 0.0136273 0.190813 0.132398 -1.7855 -0.605043 0.0243369 1 frame_000197.png

import numpy as np
import sys
import os

def quaternionToRotation(q):
    # input: [w, x, y, z]
    # source: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    R = np.array([[1-2*q[2]**2-2*q[3]**2, 2*q[1]*q[2]-2*q[3]*q[0], 2*q[1]*q[3]+2*q[2]*q[0]],
                  [2*q[1]*q[2]+2*q[3]*q[0], 1-2*q[1]**2-2*q[3]**2, 2*q[2]*q[3]-2*q[1]*q[0]],
                  [2*q[1]*q[3]-2*q[2]*q[0], 2*q[2]*q[3]+2*q[1]*q[0], 1-2*q[1]**2-2*q[2]**2]])
    return R

def convertPose(inFile, outDir):
    if not os.path.exists(inFile):
        print("Could not find file:", inFile)
        quit()
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    with open(inFile) as f:
        lines = f.readlines()

    for i in range(4, len(lines), 2):
        l = lines[i].split(" ")
        name = l[-1].split(".")[0] + ".txt"
        R = quaternionToRotation(np.array(l[1:5], dtype = float))
        outLines = [list(R[i]) + [float(l[5 + i])] for i in range(3)]
        outLines.append([0, 0, 0, 1])
        outLines = [" ".join([str(elt) for elt in line]) for line in outLines]
     
        with open(os.path.join(outDir, name), "w") as f:
            f.writelines("\n".join(outLines))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(sys.argv)
        print("Usage: convertPose.py <input file> <output directory>")
        quit()

    convertPose(sys.argv[1], sys.argv[2]) 
