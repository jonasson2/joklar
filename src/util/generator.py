import tensorflow as tf, numpy as np, albumentations, random

class AugmentDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, params=None, seed=None):
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.params = params

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        mask = self.y[idx]
        if self.params:
          aug_img, aug_mask = self._apply(image, mask)
          return (np.expand_dims(np.array(aug_img), axis=0),
                  np.expand_dims(np.array(aug_mask), axis=0))
        else:
          return (np.expand_dims(np.array(image), axis=0),
                  np.expand_dims(np.array(mask), axis=0))

    def get_albumentations_transform(self, par):
      transform = albumentations.Compose([
          albumentations.RandomGamma(
              p=par['random_gamma_probability'],
              gamma_limit=par['random_gamma_gamma_limit']),
          albumentations.VerticalFlip(p=par['flipud_probability']),
          albumentations.HorizontalFlip(p=par['fliplr_probability']),
          albumentations.RandomRotate90(p=par['rotate90_probability']),
          albumentations.RandomResizedCrop(
              p = par['random_crop_probability'],
              height = par['random_crop_height'],
              width = par['random_crop_width'],
              scale = (par['random_crop_scale_x'], par['random_crop_scale_y']))
      ], bbox_params = None)

      return transform

    def _apply(self, image, mask):
        transform = self.get_albumentations_transform(self.params)
        transformed = transform(image=image, mask=mask)

        return transformed['image'], transformed['mask']
