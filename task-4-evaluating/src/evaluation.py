from .const.evaluating_const import PROCESSED_VAL_PATH, DH_LOG_MET_PATH
from src.const.general_const import IMG_SIZE, CLASS_NAME_PATH, CLASS_MODE, BATCH_SIZE, PROD_MODEL_PATH
from src.utiles.functions import load_dataset
import tensorflow as tf
from dagshub import dagshub_logger

if __name__ == '__main__':
    # Load dataset
    test_dataset = load_dataset(PROCESSED_VAL_PATH, BATCH_SIZE, IMG_SIZE, CLASS_MODE)

    model = tf.keras.models.load_model(PROD_MODEL_PATH)

    trained_model_loss, trained_model_accuracy = model.evaluate(test_dataset)

    with dagshub_logger(should_log_hparams=False, metrics_path=DH_LOG_MET_PATH) as logger:
        metrics = {'test_set_loss': trained_model_loss, 'test_set_accuracy': trained_model_accuracy}
        logger.log_metrics(metrics)

    with open(CLASS_NAME_PATH, "r") as textfile:
        class_names = textfile.read().split(',')

    # print_data(test_dataset, class_names, notebook=NOTEBOOK, process=False, save=True,
    #            predict=True, save_path=PRED_PATH, model=model)
