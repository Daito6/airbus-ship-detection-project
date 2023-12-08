import os
import cv2
import numpy as np
import tensorflow as tf

from keras.models import load_model

from matplotlib import pyplot as plt

from defines import IMG_SCALING
from train_utils.loss_metrics import dice_coef
from paths import TEST_DIR, BASE_DIR


def prediction(test_dir, image, model):
    # Load and preprocess the test image
    rgb_path = os.path.join(test_dir, image)
    image = cv2.imread(rgb_path)[:: IMG_SCALING[0], :: IMG_SCALING[1]]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255
    image = tf.expand_dims(image, axis=0)

    # Make predictions using the model
    pred = np.squeeze(model.predict(image), axis=0)
    return cv2.imread(rgb_path), pred


# Load the pre-trained model
model = load_model(
    os.path.join(BASE_DIR, "models/model_full.h5"), custom_objects={"dice_coef": dice_coef}
)

# Commented out IPython magic to ensure Python compatibility.
test_imgs = [
    "01cafb896.jpg",
    "016e4e530.jpg",
    "009c7f8ec.jpg",
    "00c3db267.jpg",
    "00dc34840.jpg",
]

# Display images and predictions
for i in range(len(test_imgs)):
    img, pred = prediction(TEST_DIR, test_imgs[i], model)
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Image")
    fig.add_subplot(1, 2, 2)
    plt.imshow(pred, interpolation=None)
    plt.axis("off")
    plt.title("Prediction")
