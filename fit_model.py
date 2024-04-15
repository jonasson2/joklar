#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
print(sys.executable)


# In[ ]:


import os
try: # Check platform (Colab or Jupyter)
    from google.colab import drive
    drive.mount('/content/drive')
    path = "/content/drive/My Drive/joklar/"
except:
    path = os.getcwd() + "/"


# In[ ]:


import numpy as np, os, sys, time, pandas as pd, tensorflow as tf, random
start_time = time.time()
import keras
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
sys.path.append(path + "src")
from util.util import install_import
from numpy import flatnonzero as find
from osgeo import gdal


# In[ ]:


# Define model type and data to use
modeltype = "unet"
augmentation = False
dataname = "lang"
model_path = path + "results/" + modeltype + "/"
data_path = path + "data/" + dataname + "/"
os.makedirs(model_path, exist_ok=True)
os.chdir(model_path)


# In[ ]:


def load_images(nfiles = "all"):
  # Load individual image and mask files, determine glacier border tiles
  (img, mask, row, col) = np.load(data_path + "lang.npz").values()
  n_channels = img.shape[-1]
  if nfiles == "all":
    nfiles = len(mask)
  return img, mask, row, col


# In[ ]:


def get_border_indices(img, mask):
  # Load individual image and mask files, determine glacier border tiles
  ntiles = len(img)
  glacier_fraction = np.zeros(ntiles)
  meta = []
  for i in range(ntiles):
    glacier_fraction[i] = np.sum(mask[i] != 0) / mask[i].size
  is_on_border = [0.01 < g < 0.99 for g in glacier_fraction]
  count = sum(is_on_border)
  border_indices = find(is_on_border)
  return border_indices


# In[ ]:


# Import project-specific packages

# NOTES
# deeplab-v3+ is copied more or less directly from the github repository
#    github.com/david8862/tf-keras-deeplabv3p-model-set
# (the original files are in the subdirectory from_github, cf differences.txt)
#
# unet is copied from...

if modeltype == "unet":
    from models.unet import get_unet
else: # deeplab
    install_import("keras_applications")
    from models.deeplabv3p import model


# In[ ]:


# Read image data and data splits and define X and Y for training and test
# %%time
# npzfile = data_path + "images.npz"
# (img, mask, is_on_border, train, val, test) = np.load(npzfile).values()


# In[ ]:


def create_model(modeltype):
    if modeltype == "unet":
      input_img = keras.layers.Input((256, 256, 13), name='img')
      Adam_params = {"learning_rate":1e-4, "clipnorm":1.0}
      model = get_unet(input_img, n_filters=64, dropout=0.2, batchnorm=True)

    else: # deeplab
      Adam_params = {"learning_rate":1e-4}
      get_deeplab = model.get_deeplabv3p_model
      model = get_deeplab(model_type='resnet50', num_classes=1,
                          model_input_shape=(256,256),
                          output_stride=16,
                          freeze_level=0,
                          weights_path=None,
                          training=True,
                          use_subpixel=False)

    model.compile(optimizer = Adam(**Adam_params),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.save('model_first.keras')
    return model
    # NOTE: Saving weights only gives a file just as big as saving the whole model


# In[ ]:


def define_callbacks(modeltype):
    if modeltype == "unet":
      early_stopping = EarlyStopping(patience=15, verbose=1)
      reduce_LR_on_plateau = ReduceLROnPlateau(factor=0.1,
                                              patience=7,
                                              min_lr=0.00001,
                                              verbose=1),

    else: # deeplab
      early_stopping = EarlyStopping(min_delta=0.01,
                                    patience=40,
                                    verbose=1,
                                    monitor='val_loss',
                                    restore_best_weights=True)
      reduce_LR_on_plateau = ReduceLROnPlateau(factor=0.1,
                                              patience=10,
                                              min_lr=1e-12,
                                              verbose=1)
    checkpoint_best = ModelCheckpoint('model_best.keras',
                                      verbose=1,
                                      monitor="val_loss",
                                      save_weights_only=True,
                                      save_best_only=True)
    checkpoint_last = ModelCheckpoint('model_last.keras',
                                      save_weights_only=True)
    callbacks = [
      EarlyStopping(patience=60, verbose=1),
      ReduceLROnPlateau(factor=0.3, patience=10, min_lr=0.00001, verbose=1),
      checkpoint_best,
      checkpoint_last,
    ]
    return callbacks


# In[ ]:


def train_val_test_split(border_indices):
  # Define data split (training, validation, and test sets)
  seed = 42
  tf_seed = 42
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(tf_seed)
  test_size = 0.15
  train_size = 0.18  # or 0.15/(1 - test_size)
  temp, test = train_test_split(border_indices, test_size = test_size, random_state=seed)
  train, val = train_test_split(temp, test_size = train_size, random_state=seed)
  return train, val, test


# In[ ]:


def albumentations_generator(img, mask, train):
  # Implement data augmentation with the albumentations package
  augmentation = {
      "random_gamma_probability": 0.5,
      "random_gamma_gamma_limit": [80, 120],
      "flipud_probability": 0.5,
      "fliplr_probability": 0.5,
      "rotate90_probability": 0.5,
      "random_crop_probability": 0.5,
      "random_crop_height": 256,
      "random_crop_width": 256,
      "random_crop_scale_x": 0.5,
      "random_crop_scale_y": 0.5
  }
  from util.generator import AugmentDataGenerator
  train_gen = AugmentDataGenerator(img[train], mask[train], augmentation)
  return train_gen


# In[ ]:


data_path


# In[ ]:


(image, mask, row, col) = load_images()
dtype = image.dtype
ntile = len(image)
nchan = image.shape[-1]
border_indices = get_border_indices(image, mask)
nborder = len(border_indices)
print(f"{nchan} channels, {ntile} tiles, {nborder} on border, datatype: {dtype}")


# In[ ]:


(train, val, test) = train_val_test_split(border_indices)
if augmentation:
  data_input = (albumentations_generator(image, mask, train),)
else:
  data_input = (image[train], mask[train])
model = create_model(modeltype)
callbacks = define_callbacks(modeltype)


# In[ ]:


# Train (save best results, and possibly all)
do_train = True
model = tf.keras.models.load_model('model_first.keras')
if do_train:
  results = model.fit(*data_input,
                      batch_size = 8 if model=="deeplab" else 8,
                      epochs = 60,
                      callbacks = callbacks,
                      validation_data=(image[val], mask[val]))
  history = results.history
else:
  model = tf.keras.models.load_model('model_best.h5')
  history = pd.read_csv('result_history.csv')


# In[ ]:


# Plot learning curve
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 5))
plt.title("Learning curve")
plt.plot(history["loss"], label="loss")
plt.plot(history["val_loss"], label="val_loss")
plt.plot(np.argmin(history["val_loss"]),
        np.min(history["val_loss"]),
        marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();


# In[ ]:


# Compute predicted probabilites everywhere; evaluate on test
probs = model.predict(image, verbose=1)
(test_loss, test_accuracy) = model.evaluate(image[test], mask[test])
print(f'Accuracy on test: {test_accuracy}')
plt.hist(probs.ravel())


# In[ ]:


# Save the training history and model predictions
pdhistory = pd.DataFrame(history)
pdhistory.to_csv("result_history.csv")
np.savez("probs.npz", probs, test_loss, test_accuracy)


# In[ ]:


# Display running time and disconnect
end_time = time.time()
min, sec = divmod(int(end_time - start_time), 60)
print(f"Total execution time: {min}:{sec:02}")


# In[ ]:


from google.colab import runtime
runtime.unassign()

