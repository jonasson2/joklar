#!/usr/bin/env python
# coding: utf-8

# Set PATH according to platform
import os
try:
    from google.colab import drive
    drive.mount('/content/drive')
    PATH = "/content/drive/My Drive/joklar/"
except ImportError:  # Not in Colab
    host = os.environ.get('HOST')
    if host == 'makki':
        PATH = os.path.expanduser("~/drive/joklar/")
    elif host == 'elja':
        PATH = os.path.expanduser("~/joklar/")
    else:
        s = f"$HOST is {host}, it should be 'makki' or 'elja'"
        raise EnvironmentError(s)
print("Working directory path set to:", PATH)

# Imports
import numpy as np, os, sys, time, pandas as pd, tensorflow as tf, random
start_time = time.time()
import keras
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

sys.path.append(PATH + "src")
from util.util import install_import
from util.image_util import load_data, compactify
from osgeo import gdal

# Define model type and data to use
MODELTYPE = "unet"
AUGMENTATION = False
MODEL_PATH = PATH + "results/" + MODELTYPE + "/"
DATA_PATH = PATH + "data/lang/"
COMPACT = False
COMBINE_TEST_VAL = True
TRYMETA = False
os.makedirs(MODEL_PATH, exist_ok=True)
os.chdir(MODEL_PATH)

# Import project-specific packages

# NOTES
# deeplab-v3+ is copied more or less directly from the GitHub repository
#    github.com/david8862/tf-keras-deeplabv3p-model-set
# (the original files are in the subdirectory from_github, cf differences.txt)
#
# unet is copied from...
if MODELTYPE == "unet":
    from unet.unet import get_unet
else:  # deeplab
    install_import("keras_applications")
    from deeplabv3p.model import get_deeplabv3p_model

# Define data split (training, validation, and test sets)
def train_val_test_split(indices):
    seed = 41
    tf_seed = 41
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(tf_seed)
    test_size = 0.15
    train_size = 0.18  # or 0.15/(1 - test_size)
    temp, test = train_test_split(indices, test_size=test_size, random_state=seed)
    train, val = train_test_split(temp, test_size=train_size, random_state=seed)
    return train, val, test


# Load data and possibly compactify it
(image, mask, *_) = load_data(DATA_PATH + "data.npz", "border")
npixel = 256
if COMPACT:
    # Compact data by a factor of fold**3
    fold = 4
    npixel //= fold
    image = compactify(image, fold=fold)
    mask = compactify(mask, fold=fold)

# Print info
dtype = image.dtype
ntile = len(image)
nchan = image.shape[-1]
print(f"{nchan} channels, {ntile} tiles, datatype: {dtype}")
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)

# Implement data augmentation with the albumentations package
def albumentations_generator(img, mask, train):
    npixel = img.shape[1]
    augmentation = {
        "random_gamma_probability": 0.5,
        "random_gamma_gamma_limit": [80, 120],
        "flipud_probability":       0.5,
        "fliplr_probability":       0.5,
        "rotate90_probability":     0.5,
        "random_crop_probability":  0.5,
        "random_crop_height":       npixel,
        "random_crop_width":        npixel,
        "random_crop_scale_x":      0.5,
        "random_crop_scale_y":      0.5
    }
    from util.generator import AugmentDataGenerator
    train_gen = AugmentDataGenerator(img[train], mask[train], augmentation)
    return train_gen

# Create unet or deeplabv3p model
def create_model(npixel, nchan, init_lr=1e-4):
    if MODELTYPE == "unet":
        input_img = keras.layers.Input((npixel, npixel, nchan), name='img')
        model = get_unet(input_img, n_filters=16, dropout=0.3, batchnorm=True)

    else:  # deeplab
        get_deeplab = get_deeplabv3p_model
        model = get_deeplab(model_type='resnet50', num_classes=1,
                            model_input_shape=(npixel, npixel),
                            output_stride=16,
                            freeze_level=0,
                            weights_path=None,
                            training=True,
                            use_subpixel=False)

    Adam_params = {"learning_rate": init_lr, "clipnorm": 1.0}
    model.compile(optimizer=Adam(**Adam_params),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    #model.save('model_first.keras')
    return model
    # NOTE: Saving weights only gives a file just as big as saving the whole model

# Define print callback
from tensorflow.keras.callbacks import Callback
class PrintCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 3 != 0: return
        logs = logs or {}
        output = f"Epoch {epoch + 1}/{self.params['epochs']}: "
        output += ", ".join([f"{k}={v:.4f}" for k, v in logs.items()])
        print(output)
    def on_train_batch_end(self, batch, logs=None):
        return
        logs = logs or {}
        output = f"Batch {batch + 1}/{self.params['steps']}: "
        output += ", ".join([f"{k}={v:.4f}" for k, v in logs.items()])
        print(output)

# Define other callbacks
def define_callbacks(factor=0.3, patience=10, min_lr=1e-5):
    checkpoint_best = ModelCheckpoint('model_best.keras',
                                      monitor="val_loss",
                                      save_weights_only=True,
                                      save_best_only=True)
    checkpoint_last = ModelCheckpoint('model_last.keras',
                                      save_weights_only=True)
    callbacks = [
        EarlyStopping(patience=60),
        ReduceLROnPlateau(factor=factor, patience=patience, min_lr=min_lr),
        checkpoint_best,
        checkpoint_last,
        PrintCallback(),
    ]
    return callbacks

# Split data into training, validation and test
(train, val, test) = train_val_test_split(range(ntile))
if COMBINE_TEST_VAL:
    val += test
if AUGMENTATION:
    data_input = (albumentations_generator(image, mask, train),)
else:
    data_input = (image[train], mask[train])

# Define lists of meta parameters to cover
if TRYMETA:
    factors = [0.1, 0.3]
    patiences = [5, 10, 20]
    min_lrs = [1e-5, 1e-6]
    init_lrs = [1e-3, 1e-4]
    batch_sizes = [8, 32, 128]
else:
    factors = [0.3]
    patiences = [10]
    min_lrs = [1e-5]
    init_lrs = [1e-4]
    batch_sizes = [32]

# Train
for factor in factors:
    for patience in patiences:
        for min_lr in min_lrs:
            for init_lr in init_lrs:
                for batch_size in batch_sizes:
                    callbacks = define_callbacks(factor, patience, min_lr)
                    model = create_model(npixel, nchan, init_lr=init_lr)
                    # Train (save best results, and possibly all)
                    results = model.fit(*data_input,
                        verbose=0,
                        batch_size=8,
                        epochs=40,
                        callbacks=callbacks,
                        validation_data=(image[val], mask[val]))
                    print(results)
                    pass
history = results.history
len(history['loss'])

# Compute predicted probabilites everywhere; evaluate on test
probs = model.predict(image, verbose=1)
(test_loss, test_accuracy) = model.evaluate(image[test], mask[test])
print(f'Accuracy on test: {test_accuracy}')

# Plot learning curve
if host != "elja":
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
    plt.figure()
    plt.hist(probs.ravel())

# Save the training history and model predictions
pdhistory = pd.DataFrame(history)
pdhistory.to_csv("result_history.csv")
np.savez("probs.npz", probs, test_loss, test_accuracy)

# Display running time and disconnect
end_time = time.time()
min, sec = divmod(int(end_time - start_time), 60)
print(f"Total execution time: {min}:{sec:02}")


try:
    from google.colab import runtime
    runtime.unassign()
except:
    pass
