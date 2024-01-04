import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# resconstruct 16 x 15 image from 240 x 1 graysacle vector
def reconstruct_image(grayscale_vector, num, y_pred, j):

    image_data = grayscale_vector.reshape((16, 15))
    
    # plt.title(f"image {num} is mistaken for a {y_pred}, should be {j}")
    plt.imshow(image_data, cmap='gray')
    plt.axis('off')
    plt.show()

class DigitClassifier:
    
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
        decoded = autoencoder.predict(X_test_norm)
        
        # test_loss = autoencoder.evaluate(X_test, X_test)
        # print("Test Loss: ", test_loss)

        return encoded, decoded

    def visualise_encoded_images(self, decoded_train):
        
        plt.figure(figsize=(20, 40))
        for i in range(10):
            for j in range(100):
                # original image
                ax = plt.subplot(20, 10, 2 * j + 1)
                plt.imshow(self.X_test[100 * i + j].reshape(16, 15))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # reconstructed image
                ax = plt.subplot(20, 10, 2 * j + 2)
                plt.imshow(decoded_train[100 * i + j].reshape(16, 15))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()

    def svm_classify(self, X_train_encoded, y_train):
        
        # define and train SVM model
        svm_model = SVC()
        svm_model.fit(X_train_encoded, y_train)
    
        # encode test data
        X_test_encoded = self.encoder.predict(self.X_test / 6)
    
        # predict using SVM model
        y_pred = svm_model.predict(X_test_encoded)
        
        # number of inaccuracies
        print("\n\nSVM Results ~")
        print("Total number of incorrect predictions:", (y_pred != self.y_test).sum())
        for i in range(10):
            num_wrong_by_digit = (y_pred[100 * i : 100 * (i + 1)] != self.y_test[100 * i : 100 * (i + 1)]).sum()
            print(f"Number of incorrect predictions for digit {i}:", num_wrong_by_digit)
        
        for j in range(10):
            print()
            for i in range(100):
                num = 100 * j + i
                if y_pred[num] != self.y_test[num]:
                    print(f"image {num} is mistaken for a {y_pred[num]}, should be {j}")
                    reconstruct_image(self.X_test[num], num, y_pred[num], j)

classifier = DigitClassifier('../raw_train.csv', '../raw_test.csv')
X_train_encoded, X_train_decoded = classifier.autoencode()
classifier.svm_classify(X_train_encoded, classifier.y_train)

# # visualise encoded images
# classifier.visualise_encoded_images(X_train_decoded)
