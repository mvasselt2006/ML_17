# =============================================================================
# autoencoder (unsupervised learner) paired with an SVM; autoencoder is not
# completely necessary but interesting to learn about so it's included
# =============================================================================

import matplotlib.pyplot as plt
import pandas as pd
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
    
    def __init__(self, train_file, test_file):
        self.train_df = pd.read_csv(train_file)
        self.test_df = pd.read_csv(test_file)
        self.X_train = self.train_df.drop(columns=['label']).values
        self.y_train = self.train_df.iloc[:, -1].values
        self.X_test = self.test_df.drop(columns=['label']).values
        self.y_test = self.test_df.iloc[:, -1].values

    def autoencode(self):
        
        # normalise pixel values
        X_train_norm = self.X_train / 6
        X_test_norm  = self.X_test  / 6
        
        # reshape feature data frames to 16 by 15
        X_train_norm = X_train_norm.reshape((len(X_train_norm), 16 * 15))
        X_test_norm  = X_test_norm.reshape((len(X_test_norm), 16 * 15))
    
        # make autoencoder
        input_img = Input(shape=(240,))
        encoded = Dense(128, activation='relu')(input_img)
        decoded = Dense(240, activation='sigmoid')(encoded)
        
        autoencoder  = Model(input_img, decoded)
        self.encoder = Model(input_img, encoded)
        
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        autoencoder.fit(X_train_norm, X_train_norm, epochs=50, batch_size=256,
                        shuffle=True, validation_data=(X_test_norm, X_test_norm))
    
        encoded = self.encoder.predict(X_test_norm)
        # decoded = autoencoder.predict(X_test_norm)
        
        # test_loss = autoencoder.evaluate(X_test, X_test)
        # print("Test Loss: ", test_loss)
    
        return encoded

    def svm_pred(self, X_train_encoded, y_train):
        
        # define and train SVM model
        svm_model = SVC()
        svm_model.fit(X_train_encoded, y_train)
    
        # encode test data
        X_test_encoded = self.encoder.predict(self.X_test / 6)
    
        # predict using SVM model
        y_pred = svm_model.predict(X_test_encoded)
        
        return y_pred
        
    def assess_pred(self, X_train, y_train):
        
        # make preds
        y_pred = self.svm_pred(X_train, y_train)
    
        # print total number of inaccuracies
        print("\n\nSVM Results ~")
        print("Total number of incorrect predictions:", (y_pred != self.y_test).sum())
        
        # # print number of inaccuracies by digit
        # for i in range(10):
        #     num_wrong_by_digit = (y_pred[100 * i : 100 * (i + 1)] != self.y_test[100 * i : 100 * (i + 1)]).sum()
        #     print(f"Number of incorrect predictions for digit {i}:", num_wrong_by_digit)
        
        # # print explicit cases of misclassification and plot image
        # for j in range(10):
        #     print()
        #     for i in range(100):
        #         num = 100 * j + i
        #         if y_pred[num] != self.y_test[num]:
        #             print(f"image {num} is mistaken for a {y_pred[num]}, should be {j}")
        #             reconstruct_image(self.X_test[num], num, y_pred[num], j)

if __name__ == "__main__":
    
    # initalise svm classifier
    svm_classifier = svm_classifier('../raw_train.csv', '../raw_test.csv')
    
    # autoencode data
    X_train_encoded = svm_classifier.autoencode()
    
    # assess predictions
    svm_classifier.assess_pred(X_train_encoded, svm_classifier.y_train)
