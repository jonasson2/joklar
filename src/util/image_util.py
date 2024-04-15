import numpy as np
from numpy import flatnonzero as find

def get_border_indices(img, mask):
    # Load individual image and mask files, determine glacier border tiles
    ntiles = len(img)
    glacier_fraction = np.zeros(ntiles)
    meta = []
    for i in range(ntiles):
        glacier_fraction[i] = np.sum(mask[i] != 0)/mask[i].size
    is_on_border = [0.01 < g < 0.99 for g in glacier_fraction]
    count = sum(is_on_border)
    border_indices = find(is_on_border)
    return border_indices

def load_data(filename, indices="all"):
    # Load individual image and mask files, determine glacier border tiles
    data = np.load(filename)
    if indices == "border":
        I = data['border_indices']
    else:
        I = range(len(data["mask"]))
    image = data['image'][I]/10000
    mask = data['mask'][I]
    row = data['row'][I]
    col = data['col'][I]
    n_channels = image.shape[-1]
    return image, mask, row, col

import numpy as np

import numpy as np

def compactify(images, fold=4):
    """
    Compact images or boolean masks by averaging blocks of pixels for numerical data,
    or applying a threshold for boolean data (True if half or more pixels in the block are True).

    Parameters:
    - images: A NumPy array. Expected shapes are either (N, H, W, C) for images with channels,
      or (N, H, W) for grayscale images or masks without channels. The array can be of
      a numerical type or boolean.
    - fold: The factor by which to reduce the dimensions. For example, a fold of 4
      will reduce each dimension by a factor of 4.

    Returns:
    - A NumPy array of the compacted images or masks.
    """
    # Validate inputs
    if images.ndim not in [3, 4]:
        raise ValueError("Input array must be 3D or 4D.")
    if images.shape[1]%fold != 0 or images.shape[2]%fold != 0:
        raise ValueError("Image dimensions must be divisible by the fold factor.")

    # Compute the new shape for the reshaping operation based on the presence of the channel dimension
    nchan = 16//fold
    if images.ndim == 4:  # If there is a channel dimension
        new_shape = (images.shape[0]//fold,
                     images.shape[1]//fold, fold,
                     images.shape[2]//fold, fold,
                     nchan)
        reshaped_images = images[0:-1:fold, :, :, 0:nchan].reshape(new_shape)
    else:  # If there is no channel dimension
        new_shape = (images.shape[0]//fold,
                     images.shape[1]//fold, fold,
                     images.shape[2]//fold, fold)
        reshaped_images = images[0:-1:fold].reshape(new_shape)
    axis_to_reduce = (2, 4)
    if images.dtype == bool:
        # Count True values and compare to the half of the block size to decide the result
        compact_images = np.sum(reshaped_images, axis=axis_to_reduce) >= (fold**2)//2
    else:
        # Calculate the mean for numerical data
        compact_images = reshaped_images.mean(axis=axis_to_reduce)

    return compact_images
