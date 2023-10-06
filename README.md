# Pneumonia-Classification

### This is a Python3 (TensorFlow) implementation of Pneumonia Detection using chest X-ray image.

## The Dataset

[comment]: <> (Uncomment when streamlit is merged into master    ![]&#40;task_5_streamlit/images/Example.png&#41;)

The dataset comprises 5,863 frontal-view chest X-ray images organized into three folders - train, test, val. 
The folders are divided into sub-folders for each image category - Pneumonia and Normal. 
[The dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) is available on the Kaggle platform.

### Acknowledgements
- [Data](https://data.mendeley.com/datasets/rscbjbr9sj/2)
- License: CC BY 4.0
- [Citation](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)

## Prerequisites
- Python 3.8+
- TensorFlow 2.5+
- All the specified requirements in the text file

## Usage
1. Clone this repository.
2. Install requirements.txt using `pip install -r requirements.txt`.
3. Use DVC to pull the files that are stored on the DAGsHub remote storage by running `dvc pull`
4. Modify the code as you wish.
5. Run `dvc repro` to run the pipeline and train the model.

**Note:** *If you are adding/removing/moving files to different directories, it can affect the DVC pipeline, and therefore
the `dvc repro` command might not run properly.*