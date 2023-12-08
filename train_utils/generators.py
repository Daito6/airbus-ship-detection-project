import os

import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from skimage.io import imread

from defines import IMG_SCALING, AUGMENT_BRIGHTNESS
from image_mask_utils import generate_single_mask
from paths import TRAIN_DIR


def make_image_gen(in_df, batch_size = 48):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)

        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(TRAIN_DIR, c_img_id)

            c_img = imread(rgb_path)
            c_mask = np.expand_dims(
                generate_single_mask(c_masks['EncodedPixels'].values),
                -1
            )

            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]

            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= batch_size:
                yield (np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                       .astype(np.float32))
                out_rgb, out_mask = [], []


args = dict(featurewise_center=False,
            samplewise_center=False,
            rotation_range=45,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.01,
            zoom_range=[0.9, 1.25],
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="reflect",
            data_format="channels_last")

if AUGMENT_BRIGHTNESS:
    args[' brightness_range'] = [0.5, 1.5]
image_gen = ImageDataGenerator(**args)

if AUGMENT_BRIGHTNESS:
    args.pop('brightness_range')
label_gen = ImageDataGenerator(**args)


def create_aug_gen(in_gen, seed=None):
    """A generator function for augmenting images and their corresponding masks."""
    np.random.seed(
        seed
        if seed is not None
        else np.random.choice(range(1234))
    )

    for in_x, in_y in in_gen:
        seed = np.random.choice(range(1234))

        g_x = image_gen.flow(
            255 * in_x,
            batch_size=in_x.shape[0],
            seed=seed,
            shuffle=True
        )
        g_y = label_gen.flow(
            in_y,
            batch_size=in_x.shape[0],
            seed=seed,
            shuffle=True
        )

        yield next(g_x)/255.0, next(g_y)
