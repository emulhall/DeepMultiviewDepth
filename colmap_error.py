import numpy as np

# depth is a 2D numpy array
# r is kernel radius - 3x3 kernel has radius 1, 5x5 has radius 2, etc
# outlier mask is option array of bools corresponding to pixels that 
#   should be less confident
def colmapError(depth, r, outlierMask = None):
    out = np.zeros(depth.shape)
    depth_padded = np.pad(depth, r, mode="edge")
    for i in range(2*r+1):
        for j in range(2*r+1):
            if i ==r and j == r:
                continue
            out += np.abs(depth - depth_padded[i: i + depth.shape[0],
                                               j: j + depth.shape[1]])

    if outlierMask != None:
        out[outlierMask] *= 2
    out -= np.min(out)
    out /= np.max(out)
    return out

