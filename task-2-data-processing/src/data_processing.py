import os
import glob
import cv2 as cv
from src.const.general_const import IMG_SIZE
from .const.processing_const import RAW_TRAIN_PATH, RAW_TEST_PATH, RAW_VAL_PATH, BASE_PROC_DIR
from pathlib import Path

# Create list of files
RAW_TRAIN_PNEUMONIA_FILE_LIST = glob.glob(os.path.join(RAW_TRAIN_PATH,"PNEUMONIA/*"))
RAW_TRAIN_NORMAL_FILE_LIST = glob.glob(os.path.join(RAW_TRAIN_PATH,"NORMAL/*"))

RAW_TEST_PNEUMONIA_FILE_LIST = glob.glob(os.path.join(RAW_TEST_PATH,"PNEUMONIA/*"))
RAW_TEST_NORMAL_FILE_LIST = glob.glob(os.path.join(RAW_TEST_PATH,"NORMAL/*"))

RAW_VAL_PNEUMONIA_FILE_LIST = glob.glob(os.path.join(RAW_VAL_PATH,"PNEUMONIA/*"))
RAW_VAL_NORMAL_FILE_LIST = glob.glob(os.path.join(RAW_VAL_PATH,"NORMAL/*"))

# Load data
RAW_TRAIN_PNEUMONIA= [cv.resize(cv.imread(path), IMG_SIZE) for path in RAW_TRAIN_PNEUMONIA_FILE_LIST]
RAW_TRAIN_NORMAL= [cv.resize(cv.imread(path), IMG_SIZE) for path in RAW_TRAIN_NORMAL_FILE_LIST]

RAW_TEST_PNEUMONIA= [cv.resize(cv.imread(path), IMG_SIZE) for path in RAW_TEST_PNEUMONIA_FILE_LIST]
RAW_TEST_NORMAL= [cv.resize(cv.imread(path), IMG_SIZE) for path in RAW_TEST_NORMAL_FILE_LIST]

RAW_VAL_PNEUMONIA= [cv.resize(cv.imread(path), IMG_SIZE) for path in RAW_VAL_PNEUMONIA_FILE_LIST]
RAW_VAL_NORMAL= [cv.resize(cv.imread(path), IMG_SIZE) for path in RAW_VAL_NORMAL_FILE_LIST]

# Create path
Path(BASE_PROC_DIR).mkdir(parents=True, exist_ok=True)

def process_img(img_list_of_list, img_path_list_of_list, BASE_PROC_DIR):
    """
    1) Creates base directory with path ../../../task-2-data-processing/data/processed-data/*/**/
        where:
         * [train/test/val]
         ** [NORMAL/PNEUMONIA]
    2) Save processed images to file with their original name.
    """

    for IMG_LIST, IMG_PATH_LIST in zip(img_list_of_list,img_path_list_of_list):
        # (1)
        base_path_list = [BASE_PROC_DIR] + IMG_PATH_LIST[0].split('/')[-3:-1]
        Path(os.path.join(*base_path_list)).mkdir(parents=True, exist_ok=True)
        for img, path in zip(IMG_LIST, IMG_PATH_LIST):
            # (2)
            file_path_list = base_path_list + [path.split('/')[-1]]
            cv.imwrite(os.path.join(*file_path_list), img)



img_list_of_list = [RAW_TRAIN_PNEUMONIA, RAW_TRAIN_NORMAL, RAW_TEST_PNEUMONIA, RAW_TEST_NORMAL,
                    RAW_VAL_PNEUMONIA, RAW_VAL_NORMAL]
img_path_list_of_list = [RAW_TRAIN_PNEUMONIA_FILE_LIST, RAW_TRAIN_NORMAL_FILE_LIST,RAW_TEST_PNEUMONIA_FILE_LIST,
                         RAW_TEST_NORMAL_FILE_LIST, RAW_VAL_PNEUMONIA_FILE_LIST, RAW_VAL_NORMAL_FILE_LIST]
# Process the images
process_img(img_list_of_list, img_path_list_of_list, BASE_PROC_DIR)