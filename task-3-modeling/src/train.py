import tensorflow as tf

from .const.modeling_const import PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, INIT_EPOCHS, \
    LEARNING_RATE, CSV_LOG_PATH, CHECKPOINT_PATH, DH_LOG_PARAM_PATH

from src.utiles.functions import print_data, load_dataset
from src.const.general_const import NOTEBOOK, IMG_SIZE, IMG_SHAPE, CLASS_NAME_PATH,\
    CLASS_MODE, BATCH_SIZE, PROD_MODEL_PATH, NUM_CLASS
from dagshub import dagshub_logger

def augmentation_and_process_layers():
    data_augmentation_layer = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.05),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.05)
        ]
    )

    rescale_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)

    return data_augmentation_layer, rescale_layer


def build_model(data_augmentation, rescale, img_shape=IMG_SHAPE,
                learning_rate=LEARNING_RATE, num_class=NUM_CLASS):
    
    base_model = tf.keras.applications.ResNet50V2(include_top=False,
                                             weights='imagenet',
                                             input_shape=img_shape)
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(num_class, activation='sigmoid')

    inputs = tf.keras.Input(shape=img_shape)
    x = data_augmentation(inputs)
    x = rescale(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def get_callbacks(csv_logger_path, checkpoint_filepath):
    csv_logger = tf.keras.callbacks.CSVLogger(csv_logger_path)

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_accuracy')

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                          monitor='val_accuracy',
                                                          mode='max',
                                                          save_best_only=True)

    return csv_logger, early_stopping, model_checkpoint


if __name__ == '__main__':

    train_dataset = load_dataset(PROCESSED_TRAIN_PATH, BATCH_SIZE, IMG_SIZE, CLASS_MODE)
    validation_dataset = load_dataset(PROCESSED_TEST_PATH, BATCH_SIZE, IMG_SIZE, CLASS_MODE)

    class_names = train_dataset.class_names
    with open(CLASS_NAME_PATH, "w") as textfile:
        textfile.write(",".join(class_names))

    if NOTEBOOK:
        print_data(validation_dataset, class_names, notebook=NOTEBOOK, process=False, save=False, predict=False)

    data_augmentation_layer, rescale_layer = augmentation_and_process_layers()

    model = build_model(data_augmentation_layer, rescale_layer, IMG_SHAPE, LEARNING_RATE)

    csv_logger, early_stopping, model_checkpoint = get_callbacks(CSV_LOG_PATH, CHECKPOINT_PATH)

    history = model.fit(train_dataset,
                        epochs=INIT_EPOCHS,
                        validation_data=validation_dataset,
                        callbacks=[csv_logger, early_stopping, model_checkpoint])

    with dagshub_logger(should_log_metrics=False,hparams_path=DH_LOG_PARAM_PATH) as logger:
        logger.log_hyperparams(model.history.params)

    model.save(PROD_MODEL_PATH)