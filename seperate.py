from pyk4a import PyK4APlayback
import numpy as np
import cv2
from pyk4a import ImageFormat
from typing import Optional, Tuple
import os

# credit to pyk4a example
def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")

# credit to pyk4a example
def convert_to_bgra_if_required(color_format: ImageFormat, color_image):
    # examples for all possible pyk4a.ColorFormats
    if color_format == ImageFormat.COLOR_MJPG:
        color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    elif color_format == ImageFormat.COLOR_NV12:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
        # this also works and it explains how the COLOR_NV12 color color_format is stored in memory
        # h, w = color_image.shape[0:2]
        # h = h // 3 * 2
        # luminance = color_image[:h]
        # chroma = color_image[h:, :w//2]
        # color_image = cv2.cvtColorTwoPlane(luminance, chroma, cv2.COLOR_YUV2BGRA_NV12)
    elif color_format == ImageFormat.COLOR_YUY2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
    return color_image

# credit to pyk4a example
def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img


def save(playback: PyK4APlayback,file):
    count = 0
    # parent output directory
    parent = r'E:\Python\CSCI5563Homework\venv\Project\output'

    depth_out = f'{file}_depth'
    color_out = f'{file}_color'
    depthdir = os.path.join(parent,depth_out)
    colordir = os.path.join(parent,color_out)

    # make the directory if it isnt made already
    try:
        os.mkdir(depthdir)
    except:
        pass
    try:
        os.mkdir(colordir)
    except:
        pass

    while True:
        print(count)
        try:
            capture = playback.get_next_capture()
            if capture.color is not None and capture.transformed_depth is not None:
                # print(f'Color:{count}')
                cv2.imwrite(os.path.join(colordir,('Color%04d.jpg' % count)), convert_to_bgra_if_required(playback.configuration["color_format"], capture.color))
                depth = capture.transformed_depth
                depth = (depth / (np.max(depth) + 0.00001) * 255)
                # print('depth%04d.jpg' % count)
                cv2.imwrite(os.path.join(depthdir,('Depth%04d.jpg' % count)), depth)
                count = count + 1

        except EOFError:
            break


def main() -> None:
    offset = 0

    # input file name
    file = str('bees2')
    playback = PyK4APlayback(rf'{file}.mkv')
    playback.open()
    # playback.save_calibration_json(r'E:\Python\CSCI5563Homework\venv\Project\calibration.json')

    info(playback)

    if offset != 0.0:
        playback.seek(int(offset * 1000000))
    save(playback,file)

    playback.close()



if __name__ == "__main__":
    main()