import os

NOTEBOOK=False

# TODO: Enlarge the image size
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
COLOR_MODE = 'rgb'
BATCH_SIZE = 32

# Dataset params
CLASS_MODE = 'binary'
NUM_CLASS = 1

BASE_PATH = os.getcwd()
CLASS_NAME_PATH = os.path.join(BASE_PATH,"task-3-modeling/src/const/class_names.txt")
PROD_MODEL_PATH = os.path.join(BASE_PATH,'task_5_streamlit/model')
