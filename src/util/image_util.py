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
