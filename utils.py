import skimage
import skimage.io
import skimage.transform
import numpy as np


# returns image of shape [1, 224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.repeat(img, 3, axis=2)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    resized_img = resized_img[np.newaxis, :, :, :]
    return resized_img
