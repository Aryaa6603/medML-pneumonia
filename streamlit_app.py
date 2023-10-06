# todo: Edit SHORT_DESCRIPTION
# todo: Change val set to be traced by git & change img path here
# todo: add docstrings to functions

# todo general:
#  - Add wights to the model,
#  - change base path in const to general const

import streamlit as st
import cv2 as cv
from PIL import Image
import numpy as np
import tensorflow as tf
from src.const.general_const import PROD_MODEL_PATH, IMG_SIZE, CLASS_NAME_PATH
from task_5_streamlit.src.const.streamlit_const import \
    DAGSHUB_IMAGE_PATH, HEALTHY_IMAGE_ONE_PATH,HEALTHY_IMAGE_TWO_PATH, SICK_IMAGE_ONE_PATH, SICK_IMAGE_TWO_PATH,HEADER,\
    SUB_HEADER, SHORT_DESCRIPTION, IMAGE_POOL_DESCRIPTION, SELECT_BOX_TEXT, SUPPORTED_IMG_TYPE, WARNING_MSG, SUCCESS_MSG, \
    BIG_FONT, MID_FONT, SMALL_FONT

def markdown_format(font_size,content):
    st.markdown(f"<{font_size} style='text-align: center;'>{content}</{font_size}>",
                unsafe_allow_html=True)

def load_n_resize_image(image_path):
    pil_img = Image.open(image_path)
    return cv.resize(cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR), IMG_SIZE)


def load_image_pool():
    healthy = [load_n_resize_image(HEALTHY_IMAGE_ONE_PATH), load_n_resize_image(HEALTHY_IMAGE_TWO_PATH)]
    sick = [load_n_resize_image(SICK_IMAGE_ONE_PATH), load_n_resize_image(SICK_IMAGE_TWO_PATH)]
    return {'healthy': healthy, 'sick': sick}


def present_pool(col, col_name, img_list):
    name_list = []
    for row in range(len(img_list)):
        col.image(img_list[row], use_column_width=True, caption=col_name + f" {row + 1}")
        name_list.append(col_name + f" {row + 1}")
    return name_list


def display_prediction(pred):
    if pred == 'sick':
        st.warning(WARNING_MSG)
    else:
        st.success(SUCCESS_MSG)


@st.cache(suppress_st_warning=True)
def get_prediction(img):
    with open(CLASS_NAME_PATH, "r") as textfile:
        class_names = textfile.read().split(',')

    img_expand = np.expand_dims(img, 0)

    model = tf.keras.models.load_model(PROD_MODEL_PATH)
    predictions = model.predict(img_expand)
    display_prediction(class_names[np.rint(predictions[0][0]).astype(int)])


def predict_for_selectbox(selectbox, my_bar, latest_iteration):
    img_class = selectbox.split()[0]
    img_position = int(selectbox.split()[-1]) - 1
    img = dict_of_img_lists[img_class][img_position]
    my_bar.progress(50)

    latest_iteration.text('Processing image')
    get_prediction(img)
    my_bar.progress(100)


def predict_for_file_buffer(file_buffer, my_bar, latest_iteration):
    latest_iteration.text('Loading image')
    img = load_n_resize_image(file_buffer)
    markdown_format(MID_FONT, "Your chest X-ray")
    st.image(img, use_column_width=True)
    my_bar.progress(50)

    latest_iteration.text('Processing image')
    get_prediction(img)
    my_bar.progress(100)



if __name__ == '__main__':
    # Page configuration
    st.set_page_config(page_title=HEADER, page_icon="🤒",
                       initial_sidebar_state='expanded')

    # Base Design
    st.image(image=DAGSHUB_IMAGE_PATH)
    markdown_format(BIG_FONT, HEADER)
    markdown_format(MID_FONT, SUB_HEADER)
    markdown_format(SMALL_FONT, SHORT_DESCRIPTION)
    latest_iteration = st.empty()
    my_bar = st.progress(0)

    # Show pool of images
    dict_of_img_lists = load_image_pool()
    with st.beta_expander("Image Pool"):
        markdown_format(MID_FONT, IMAGE_POOL_DESCRIPTION)
        col1, col2 = st.beta_columns(2)
        healthy_sidebar_list = present_pool(col1, "healthy", dict_of_img_lists['healthy'])
        sick_sidebar_list = present_pool(col2, "sick", dict_of_img_lists['sick'])

    # Sidebar
    selectbox = st.sidebar.selectbox(SELECT_BOX_TEXT,
                                     [None] + healthy_sidebar_list + sick_sidebar_list)

    file_buffer = st.sidebar.file_uploader("", type=SUPPORTED_IMG_TYPE)

    # Predict for user selection
    if selectbox:
        predict_for_selectbox(selectbox, my_bar, latest_iteration)
        dict_of_img_lists = load_image_pool()

    if file_buffer:
        predict_for_file_buffer(file_buffer, my_bar, latest_iteration)

