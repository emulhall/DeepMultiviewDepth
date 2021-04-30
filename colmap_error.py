import numpy as np
from scipy import signal

# depth is a 2D numpy array
# r is kernel radius - 3x3 kernel has radius 1, 5x5 has radius 2, etc
# outlier mask is option array of bools corresponding to pixels that 
#   should be less confident
def colmapError(depth, r, outlierMask):
    out = np.zeros(depth.shape)
    depth_padded = np.pad(depth, r, mode="edge")
    for i in range(2*r+1):
        for j in range(2*r+1):
            if i ==r and j == r:
                continue
            out += np.abs(depth - depth_padded[i: i + depth.shape[0],
                                               j: j + depth.shape[1]])
    out=np.where(depth>0,0.25,0.75)
    out[outlierMask] *= 2
    out -= np.min(out)
    if np.max(out) != 0:
        out /= np.max(out)
    return out

def edgeDetectionError(depth, r, outlierMask, grayscale):
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype='float')
    edge = signal.convolve2d(grayscale, kernel, mode="same", boundary="fill")
    edge -= np.min(edge)
    if np.max(edge) != 0:
        edge /= np.max(edge)

    colmap = colmapError(depth, r, outlierMask)
    error = colmap - edge
    error[error < 0] = 0
    return error
