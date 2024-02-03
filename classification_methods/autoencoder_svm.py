# =============================================================================
# autoencoder (unsupervised learner) paired with an SVM; autoencoder is not
# completely necessary but interesting to learn about so it's included
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


# resconstruct 16 x 15 image from 240 x 1 graysacle vector
def reconstruct_image(image, num, y_pred, j):

    reshaped_image = image.reshape((16, 15))
    
    plt.title(f"image {num} is mistaken for a {y_pred}, should be {j}")
    plt.imshow(reshaped_image, cmap='gray')
    plt.axis('off')
    plt.show()

class svm_classifier:
    
    def __init__(self, X_train, y_train, X_test, y_test, param):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # hyperparameters for fine tuning
        self.batchsize = param

    def autoencode(self):
        
        # normalise pixel values
        X_train = self.X_train / 6
        X_test  = self.X_test  / 6
        
        # hyperparameters
        enc_layers = 64
        dec_layers = 240
        
        enc_act_func = 'relu'
        dec_act_func = 'sigmoid'
        
        auto_enc_opt    = 'nadam'
        auto_enc_loss   = 'mean_squared_error'
        auto_enc_epochs = 97
        auto_enc_batchs = 108
    
        # make autoencoder
        input_layer = Input(shape=(240,))
        encoded = Dense(enc_layers, activation=enc_act_func)(input_layer)
        decoded = Dense(dec_layers, activation=dec_act_func)(encoded)
        
        autoencoder  = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)
        
        autoencoder.compile(optimizer=auto_enc_opt, loss=auto_enc_loss)
        autoencoder.fit(X_train, X_train, epochs=auto_enc_epochs,
                        batch_size=auto_enc_batchs, shuffle=True,
                        validation_data=(X_test, X_test), verbose=0)

        encoded = self.encoder.predict(X_test, verbose=0)
        decoded = autoencoder.predict(X_test, verbose=0)
        
        # test_loss = autoencoder.evaluate(X_test, X_test)
        # print("Test Loss: ", test_loss)
    
        return encoded

    def svm_pred(self, X_train_enc, y_train):
        
        # define and train SVM model
        # hyperparameteres here are choice of kernel and regularisation 'C'
        svm_model = SVC(kernel='rbf', C=1.2)
        svm_model.fit(X_train_enc, y_train)
    
        # encode test data
        X_test_encoded = self.encoder.predict(self.X_test / 6, verbose=0)

        # predict using SVM model
        y_pred = svm_model.predict(X_test_encoded)

        return y_pred
    
    def confusion_matrix(self, y_test, y_pred):
        
        conf_mat = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="coolwarm", xticklabels=range(10), yticklabels=range(10))
        plt.title("Autoencoding-paired SVM confusion matrix")
        plt.ylabel('Correct Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
    def assess_pred(self, X_train, y_train):
        
        # make predictions
        y_pred = self.svm_pred(X_train, y_train)
    
        # print total number of inaccuracies
        print("\nSVM Results ~")
        print("Total number of incorrect predictions:", (y_pred != self.y_test).sum())
        print()
        
        # print number of inaccuracies by digit
        for i in range(10):
            num_wrong_by_digit = (y_pred[100 * i : 100 * (i + 1)] != self.y_test[100 * i : 100 * (i + 1)]).sum()
            print(f"Number of incorrect predictions for digit {i}:", num_wrong_by_digit)
        print()
        
        # print explicit cases of misclassification and plot image
        for j in range(10):
            for i in range(100):
                num = 100 * j + i
                if y_pred[num] != self.y_test[num]:
                    print(f"image {num} is mistaken for a {y_pred[num]}, should be {j}")
                    reconstruct_image(self.X_test[num], num, y_pred[num], j)
                    
        self.confusion_matrix(y_test, y_pred)
                    
        # return (y_pred != self.y_test).sum()
        
def cross_validate(folds, X_train, y_train, X_test, y_test, param):
    
    skf = StratifiedKFold(n_splits=folds)
    for epochs in range(32, param + 1):
        
        fold_num_wrong = []
        for train_index, val_index in skf.split(X_train, y_train):
            
            # split data into training and validation folds
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            classifier = svm_classifier(X_train_fold, y_train_fold,
                                        X_val_fold, y_val_fold, epochs)
            X_train_enc = classifier.autoencode()
            fold_num_wrong.append(classifier.assess_pred(X_train_enc,
                                                         y_val_fold))
        print(np.mean(fold_num_wrong), epochs)
    
if __name__ == "__main__":
    
    train_df = pd.read_csv('../raw_train.csv')
    test_df = pd.read_csv('../raw_test.csv')
    
    # =============================================================================
    # COMMENT OUT SHUFFLING (LINES 160 AND 161) IF USING ASSESS PRED FULLY 
    # =============================================================================
    
    # shuffle data so that it doesn't try to learn order of digits
    # this is somewhat out of paranoia but it doesn't hurt to do
    # train_df = train_df.sample(frac=1)
    # test_df  = test_df.sample(frac=1)
    
    X_train = train_df.drop(columns=['label']).values
    y_train = train_df.iloc[:, -1].values
    X_test  = test_df.drop(columns=['label']).values
    y_test  = test_df.iloc[:, -1].values
    
    batch_size_max = 128
    
    # # ten-fold cross-validation for tuning epochs/batch_size_max
    # cross_validate(10, X_train, y_train, X_test, y_test, batch_size_max)
    
    # individual run
    classifier = svm_classifier(X_train, y_train,
                                X_test, y_test, batch_size_max)
    X_train_enc = classifier.autoencode()
    classifier.assess_pred(X_train_enc, y_test)
    
    # # average misclassification rate of 100 runs
    # avg = 0
    # for i in range(100):
    #     classifier = svm_classifier(X_train, y_train,
    #                                 X_test, y_test, batch_size_max)
    #     X_train_enc = classifier.autoencode()
    #     avg += classifier.assess_pred(X_train_enc, y_test)
    # print(avg/100)
        