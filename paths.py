import os

BASE_DIR = "/Users/macbookpro/Documents/CompVision/DataSet"

TRAIN_DIR = BASE_DIR + "/train_v2/"
TEST_DIR = BASE_DIR + "/test_v2/"

TRAIN = os.listdir(TRAIN_DIR)
TEST = os.listdir(TEST_DIR)
