import os

# Data path
BASE_PATH = os.getcwd()
PROCESSED_TRAIN_PATH = os.path.join(BASE_PATH,'task-2-data-processing/data/processed-data/train')
PROCESSED_TEST_PATH = os.path.join(BASE_PATH,'task-2-data-processing/data/processed-data/test')

# Model params
INIT_EPOCHS = 1
LEARNING_RATE = 1e-6

# Save path
CSV_LOG_PATH = os.path.join(BASE_PATH,'task-3-modeling/model-checkpoint/TF_training_logs.csv')
CHECKPOINT_PATH = os.path.join(BASE_PATH,'task-3-modeling/model-checkpoint/TF-Model-Checkpoint/')
DH_LOG_PARAM_PATH = os.path.join(BASE_PATH,'task-4-evaluating/eval/params/params.yml')
