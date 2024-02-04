# Handwritten digit classification

## Note: the data sets included here are small. No need to worry about the inclusion of bloated data sets.

## Overview:
This repository contains implementations of kNN, Random Forest, CNN and autoencoding-paired SVM approaches to a multiclass classification task pertaining to handwritten digit.

## Dependencies:
To use these programs, you need Python installed on your machine along with a series of libraries including:
- NumPy
- Pandas
- scikit-learn

Note that there are far more libraries than only these. Each program begins by indicating the dependencies, if any.

Installing each library can be done using pip by running `pip install <name of library>`.

## How to run:
First, ensure that the path of your Python environment is that of the root directory of this repository. Each classification method is contained in the `classification_methods` directory.

`....py`: ?

`....py`: ...

## Misc:
- The augmentation methods used are contained in `data_augmentation/aug_methods.py` for which illustrative examples are provided in `data_augmentation/aug_examples.py`.

- The handcrafted features used are contained in the `handcrafted_features` directory. Most include an `if __name__=="__main__"` component illustrating its use and the corresponding output.

- `features_to_csv.py` is used to produce a CSV file whose columns are the handcrafted features applied to the raw data (both training and testing).