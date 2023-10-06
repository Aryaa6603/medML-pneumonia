import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from src.const.general_const import COLOR_MODE

def load_dataset(data_path, batch_size, img_size, label_mode):
    return image_dataset_from_directory(data_path, shuffle=True, batch_size=batch_size,
                                        image_size=img_size, label_mode=label_mode,
                                        color_mode=COLOR_MODE)


def remove_img_in_dir(save_path):
    base_path = os.path.join(os.getcwd(), save_path)
    for f in glob.glob(os.path.join(base_path, "*.png")):
        os.remove(f)


def print_data(dataset, class_names, notebook, process=False, save=False, predict=False, **kwargs):
    if save:
        remove_img_in_dir(kwargs['save_path'])

    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            img_name = class_names[np.argmax(labels[i])]
            img = images[i].numpy().astype("uint8")

            if process:
                augmented_image = kwargs['rescale_layer'](kwargs['data_augmentation_layer'](tf.expand_dims(img, 0)))
                img = augmented_image[0]
                img_name = img_name + ' ' + str(i)

            if predict:
                predictions = kwargs['model'].predict(dataset)
                pred = class_names[np.argmax(predictions[i])]
                score = np.max(predictions[i]) * 100
                img_name = "Real: " + img_name + ", Pred: " + pred + "\n,Confidence: " + str(round(score, 2))

            if notebook:
                plt.imshow(img)

            plt.title(img_name)
            plt.axis("off")
        if save:
            plt.tight_layout()
            plt.savefig(os.path.join(*[os.getcwd(), kwargs['save_path'], 'All_Plots.png']))
