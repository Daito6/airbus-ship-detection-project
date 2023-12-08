"""Constants for data preparation and augmentation"""

# Number of samples per group for dataset balancing
SAMPLES_PER_GROUP = 4000

# Alpha parameter for loss function
ALPHA = 0.8

# Gamma parameter for loss function
GAMMA = 2

# Number of images for validation
VALID_IMG_COUNT = 600

# Down sampling factor for image preprocessing
IMG_SCALING = (3, 3)     

# Flag to enable brightness augmentation (True/False)
AUGMENT_BRIGHTNESS = False
