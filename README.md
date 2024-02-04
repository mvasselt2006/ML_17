# Handwritten digit classification

## Note: the data sets included here are small. No need to worry about the inclusion of bloated data sets.

## Overview:
This repository contains implementations of kNN, Random Forest, CNN and autoencoding-paired SVM approaches to a multiclass classification task pertaining to handwritten digit.

## Dependencies:
To use these programs, you need Python installed on your machine along with a series of libraries including:
- NumPy
- Pandas
- scikit-learn
- matplotlib
- TensorFlow

Note that there are far more libraries than only these. Each program begins by indicating the dependencies, if any.

Installing each library can be done using pip by running `pip install <name of library>`.

## How to run:
First, ensure that the path of your Python environment is that of the root directory of this repository. Each classification method is contained in the `classification_methods` directory.

`....py`: ?

- `CNN.py`:
Dependancies on: TensorFlow, scikit-learn along with a few standard ones

The code imports the raw_test.cvs and raw_train.cvs files assuming they can be found in the same directory. Run the command "training(train_dat,train_label, test_dat, test_label,epoch)" to train a model model or tf.keras.models.load_model('model.keras') to load an already trained model. The best performing model has been uploaded in the same folder as the CNN.py file under the name "model3.keras". get_cfmatrix(labels,images,model) can be run to obtain a the confusion matrix for a given model and print_misclass(model,x_size, y_size) will print all the missclassified digits. Most information on how to use the functions can be found in the file itself

## Misc:
- The augmentation methods used are contained in `data_augmentation/aug_methods.py` for which illustrative examples are provided in `data_augmentation/aug_examples.py`.

- The handcrafted features used are contained in the `handcrafted_features` directory. Most include an `if __name__=="__main__"` component illustrating its use and the corresponding output.

- `features_to_csv.py` is used to produce a CSV file whose columns are the handcrafted features applied to the raw data (both training and testing).
