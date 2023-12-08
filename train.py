# -*- coding: utf-8 -*-
import pandas as pd
import gc

from sklearn.model_selection import train_test_split

from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from keras.optimizers import Adam

from train_utils.generators import make_image_gen, create_aug_gen
from train_utils.image_mask_utils import prepare_data
from defines import (
    SAMPLES_PER_GROUP,
    VALID_IMG_COUNT,
)
from train_utils.loss_metrics import dice_coef, Loss
from paths import BASE_DIR
from train_utils.unet import create_unet

masks, unique_img_ids = prepare_data(BASE_DIR)

# Let's balance the dataset by randomly selecting SAMPLES_PER_GROUP elements
# for each category (groups based on the number of ships) to avoid imbalance
# between categories.
balanced_train_df = (
    unique_img_ids.groupby("ships")
    .apply(
        lambda x: x.sample(SAMPLES_PER_GROUP)
        if len(x) > SAMPLES_PER_GROUP else x
    )
)


# Split dataset into training and validation sets
train_ids, valid_ids = train_test_split(
    balanced_train_df,
    test_size=0.2,
    stratify=balanced_train_df["ships"]
)

train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)

print(train_df.shape[0], "training masks")
print(valid_df.shape[0], "validation masks")

# Data generation for training and validation
train_gen = make_image_gen(train_df)
train_x, train_y = next(train_gen)
valid_x, valid_y = next(make_image_gen(valid_df, VALID_IMG_COUNT))

# Augmented data generation for training
cur_gen = create_aug_gen(train_gen)
t_x, t_y = next(cur_gen)

# Memory management: garbage collector activation and execution
gc.enable()
gc.collect()

# U-Net model initialization
model = create_unet()

# Callbacks setting
weight_path = "models/{}_weights.hdf5".format("model")

checkpoint = ModelCheckpoint(
    weight_path,
    monitor="val_dice_coef",
    verbose=1,
    save_best_only=True,
    mode="max",
    save_weights_only=True
)

early = EarlyStopping(monitor="val_dice_coef", mode="max", patience=15)

reduceLROnPlat = ReduceLROnPlateau(
    monitor="val_dice_coef",
    factor=0.5,
    patience=3,
    verbose=1,
    mode="max",
    epsilon=0.0001,
    cooldown=2,
    min_lr=1e-6
)

callbacks_list = [checkpoint, early, reduceLROnPlat]

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=Loss,
    metrics=[dice_coef, "binary_accuracy"]
)

step_count = min(5, train_df.shape[0]//48)
aug_gen = create_aug_gen(make_image_gen(train_df))

model.fit(
    aug_gen,
    steps_per_epoch=step_count,
    epochs=15,
    validation_data=(valid_x, valid_y),
    callbacks=callbacks_list,
    workers=1
)

model_v2 = model
model_v2.save("models/model_full.h5")
