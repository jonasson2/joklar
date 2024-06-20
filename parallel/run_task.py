#!/usr/bin/env python

import log

# External imports
log.debug('External imports in run_task.py')
import numpy as np, os, sys, time, json, tensorflow as tf, random, keras
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import KFold, train_test_split
log.debug('Finished external imports')
# Local inmports
from par_util import get_paths
src = get_paths()['src']
sys.path.append(src)
from util.util import install_import
from util.image_util import load_data, compactify
import log

COMPACT = True
COMBINE_TEST_VAL = False
NSPLIT = 2
SEED = 42
TESTSIZE = 0.15
EPOCHS = 60

def test_split(indices, seed=SEED):
    # Define data split (training, validation, and test sets)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    nontest, test = train_test_split(indices, test_size=TESTSIZE,
                                     random_state=seed)
    return nontest, test

def albumentations_generator(img, mask, train):
    # Implement data augmentation with the albumentations package
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

def create_model(modeltype, npixel, nchan, init_lr=1e-4):
    # Create unet or deeplabv3p model
    if modeltype == "unet":
        from unet.unet import get_unet
        input_img = keras.layers.Input((npixel, npixel, nchan), name='img')
        model = get_unet(input_img, n_filters=16, dropout=0.3, batchnorm=True)
    else:  # deeplab
        install_import("keras_applications")
        from deeplabv3p.model import get_deeplabv3p_model
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
    # NOTE:
    # Saving weights only gives a file just as big as saving the whole model
    return model

class PrintCallback(Callback):
    # Define print callback
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 3 != 0: return
        logs = logs or {}
        output = f"Epoch {epoch + 1}/{EPOCHS}: "
        output += ", ".join([f"{k}={v:.4f}" for k, v in logs.items()])
        log.info(output)
    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        output = f"Batch {batch + 1}/{self.params['steps']}: "
        output += ", ".join([f"{k}={v:.4f}" for k, v in logs.items()])

def define_callbacks(factor=0.3, patience=10, min_lr=1e-5):
    # Define other callbacks
    checkpoint_best = ModelCheckpoint('model_best.keras',
                                      monitor="val_loss",
                                      save_weights_only=True,
                                      save_best_only=True)
    checkpoint_last = ModelCheckpoint('model_last.keras',
                                      save_weights_only=True)
    callbacks = [
        # EarlyStopping(patience=60),
        ReduceLROnPlateau(factor=factor, patience=patience, min_lr=min_lr),
        checkpoint_best,
        checkpoint_last,
        PrintCallback(),
    ]
    return callbacks

class TrainingMeasures:
    def __init__(self):
        self.optimal_loss = []
        self.optimal_epoch = []
        self.accuracy = []
        self.training_time = []
        self.loss_curve = []
        self.val_loss_curve = []

    def __repr__(self):
        attrs = vars(self)
        # Creating an indented string for each attribute
        attrs_str = '\n  '.join(f"{k}={v!r}" for k, v in attrs.items())
        return f"{self.__class__.__name__}:\n  {attrs_str}"

    def update(self, history, training_time):
        # Update scalar values
        loss = history['loss']
        val_loss = history['val_loss']
        optimal_epoch = np.argmin(val_loss)
        optimal_loss = val_loss[optimal_epoch]
        accuracy = history['val_accuracy'][optimal_epoch]

        self.optimal_loss.append(optimal_loss)
        self.optimal_epoch.append(optimal_epoch)
        self.accuracy.append(accuracy)
        self.training_time.append(training_time)
        self.loss_curve.append(loss)
        self.val_loss_curve.append(val_loss)
        
    def stats(self, numdec=4):
        mean_SD = {}
        for attr in vars(self):
            values = getattr(self, attr)
            mean_SD[attr] = (np.round(np.mean(values, axis=0), numdec).tolist(),
                             np.round(np.std(values, axis=0), numdec).tolist())
        return mean_SD

def fit_model(task_number, params, datapath):
    # Define model type and data to use, set parameters
    modeltype = params["modeltype"]
    augment = params["augment"]
    factor = params["factor"]
    patience = params["patience"]
    min_lr = params["min_lr"]
    init_lr = params["init_lr"]
    batch_size = params["batch_size"]
    
    # Load data and possibly compactify it
    (image, mask, *_) = load_data(datapath + "/data.npz", "border")
    npixel = 256
    if COMPACT:
        # Compact data by a factor of fold**3
        fold = 4
        npixel //= fold
        image = compactify(image, fold=fold)
        mask = compactify(mask, fold=fold)

    # Log info
    dtype = image.dtype
    ntile = len(image)
    nchan = image.shape[-1]
    log.info(f"{nchan} channels, {ntile} tiles, datatype: {dtype}")
    log.info(f"Image shape: {image.shape}")
    log.info(f"Mask shape: {mask.shape}")

    # Train-test split
    nontest, test = test_split(range(ntile), SEED)
    
    # Use k-fold cross-validation
    kf = KFold(n_splits=NSPLIT, shuffle=True, random_state=SEED)
    measures = TrainingMeasures()

    for fold, (train_idx, val_idx) in enumerate(kf.split(nontest)):
        log.info(f"Training fold {fold+1}/{NSPLIT}")
        
        if augment:
            data_input = (albumentations_generator(image, mask, train_idx),)
        else:
            data_input = (image[train_idx], mask[train_idx])
        model = create_model(modeltype, npixel, nchan, init_lr=init_lr)
        
        callbacks = define_callbacks(factor, patience, min_lr)
        start_time = time.time()
        results = model.fit(*data_input,
                            batch_size=batch_size,
                            epochs=EPOCHS,
                            callbacks=callbacks,
                            validation_data=(image[val_idx], mask[val_idx]),
                            verbose=0)
        train_minutes = (time.time() - start_time)/60
        measures.update(results.history, train_minutes)
        clear_session()
        
    stats = measures.stats()
    outfile = "measures.json"
    with open(outfile, 'w') as f:
        json.dump([params, stats], f, indent=2)
