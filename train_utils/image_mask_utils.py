import os

import numpy as np
import pandas as pd


def prepare_data(data_path):
    masks = pd.read_csv(
        os.path.join(data_path, 'train_ship_segmentations_v2.csv')
    )
    not_empty = pd.notna(masks.EncodedPixels)

    print(not_empty.sum(), 'masks in',
          masks[not_empty].ImageId.nunique(), 'images')
    print((~not_empty).sum(), 'empty images in',
          masks.ImageId.nunique(), 'total images')

    masks['ships'] = (masks['EncodedPixels']
                      .map(lambda c_row: 1 if isinstance(c_row, str) else 0))
    unique_img_ids = (masks.groupby('ImageId')
                      .agg({'ships': 'sum'}).reset_index())
    unique_img_ids['has_ship'] = (unique_img_ids['ships']
                                  .map(lambda x: 1.0 if x > 0 else 0.0))

    masks.drop(['ships'], axis=1, inplace=True)

    return masks, unique_img_ids


def decode_rle(encoded_mask, image_shape=(768, 768)):
    """Function for decoding the RLE representation of a mask in an image."""
    components = encoded_mask.split()

    start_indices, lengths = [
        np.asarray(x, dtype=int)
        for x in (components[0:][::2], components[1:][::2])
    ]
    start_indices -= 1
    end_indices = start_indices + lengths

    decoded_mask = np.zeros(image_shape[0] * image_shape[1], dtype=np.uint8)

    for start, end in zip(start_indices, end_indices):
        decoded_mask[start:end] = 1

    return decoded_mask.reshape(image_shape).T


def generate_single_mask(mask_list):
    """Function to combine individual ship masks into a single array"""
    all_masks_array = np.zeros((768, 768), dtype=np.uint8)

    for individual_mask in mask_list:
        if isinstance(individual_mask, str):
            all_masks_array |= decode_rle(individual_mask)

    return all_masks_array
