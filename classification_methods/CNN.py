import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

# load data frames
train_df = pd.read_csv("raw_train.csv")
test_df = pd.read_csv("raw_test.csv")

# split training data into features and label
train_images = train_df.iloc[:, :-1]
train_labels = train_df.iloc[:, -1]

# split testing data into features and label
test_images = test_df.iloc[:, :-1]
test_labels = test_df.iloc[:, -1]

class_names = [0,1,2,3,4,5,6,7,8,9]

# =====================================
# Plotting images
# =====================================

# plt.figure(figsize=(10,10))
# for i in range(10):
#     plt.subplot(5,2,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     img= np.array(train_images.iloc[100*i,:])
#     print(img)
#     plt.imshow(img.reshape((16,15)))
#     plt.xlabel(class_names[train_labels[i*100]])
# plt.show()

#make images into correct shape
train_img= np.zeros((1000,16,15))
test_img = np.zeros((1000,16,15))
for i in range(1000):
    train_img[i,:,:] = np.array(train_images.iloc[i,:]).reshape(16,15)
    test_img[i,:,:] = np.array(test_images.iloc[i,:]).reshape(16,15)

def CNN1():
    """"Basic test"""
    CNN = models.Sequential()
    CNN.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(16, 15, 1)))
    CNN.add(layers.MaxPooling2D((2, 2)))
    CNN.add(layers.Conv2D(64, (3, 3), activation='relu'))
    CNN.add(layers.MaxPooling2D((2, 2)))

    CNN.add(layers.Flatten())
    CNN.add(layers.Dense(64, activation='relu'))
    CNN.add(layers.Dense(10))
    #CNN.summary()

    CNN.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return CNN

def CNN2():
    """Model that returned the best results"""
    CNN = models.Sequential()
    CNN.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(16, 15, 1), padding='same'))
    CNN.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(16, 15, 1), padding='same'))
    CNN.add(layers.MaxPooling2D((2, 2)))
    CNN.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same'))
    CNN.add(layers.MaxPooling2D((2, 2)))
    CNN.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same'))

    CNN.add(layers.Flatten())
    CNN.add(layers.Dense(10, activation='softmax'))

    CNN.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'))
    CNN.summary()
    return CNN

def CNN3():
    CNN = models.Sequential()
    CNN.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(16, 15, 1), padding='same'))
    CNN.add(layers.Conv2D(32, (7, 7), activation='relu', padding='same'))
    CNN.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same'))
    CNN.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same'))
    CNN.add(layers.MaxPooling2D((2, 2)))
    CNN.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same'))

    CNN.add(layers.Flatten())
    CNN.add(layers.Dense(10, activation='softmax'))

    CNN.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'))
    CNN.summary()
    return CNN


def training(train_dat,train_label, test_dat, test_label,epoch):
    """Trains the model with the training data. To change which model to run this
    for, change the first line of this function to the wanted CNN model"""
    model= CNN2()
    history = model.fit(train_dat, train_label, epochs=epoch, 
                        validation_data=(test_dat, test_label))
    #model.save("Model4.keras")
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()
    test_loss, test_acc = model.evaluate(test_dat,  test_label, verbose=2)
    print(f"Model accuracy: {test_acc}")
    return model


def get_cfmatrix(labels,images,model):
    """Plots the confusion matrix for the model compared to the test labels"""
    pred_labels = model.predict(np.reshape(images,(1000,16,15,1)))
    pred = (np.argmax(pred_labels, axis=1)).flatten()
   
    cf_matrix = metrics.confusion_matrix(labels, pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cf_matrix)
    disp.plot()
    plt.show()

def shuffle_data(train_dat,train_label,test_dat,test_label):
    """"Shuffles the dataset and splits the train and test data randomly"""
    pics = np.concatenate(train_dat, test_dat)
    labels= np.concatenate(train_label,test_label)
    train_pics, test_pics, train_labels, test_labels = train_test_split(pics, labels, test_size = 0.5, shuffle=True)
    return train_pics, test_pics, train_labels, test_labels

def print_misclass(model,x_size, y_size):
    """Plots all the incorrectely labeled test images for the model. These images are plotted on a 
    grid of (x_size, y_size) images"""
    pred_labels = model.predict(np.reshape(test_img,(1000,16,15,1)))
    pred = (np.argmax(pred_labels, axis=1)).flatten()
    i_incorrect=[]
    for i in range(1000):
        if pred[i]-test_labels[i] != 0:
            i_incorrect.append(i)

    print(i_incorrect)
    plt.figure(figsize=(20,15))
    for i in range(len(i_incorrect)):
        plt.subplot(x_size,y_size,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        ind= i_incorrect[i]
        img= np.array(test_images.iloc[ind,:])
        plt.imshow(img.reshape((16,15)),cmap='gray')
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        plt.xlabel(f"True {test_labels[ind]}, pred {pred[ind]}")
    plt.show()


#Trains a new model
#mod= training(train_img, train_labels, test_img, test_labels,15)
#Loads a trained model
mod = tf.keras.models.load_model('model3.keras')

#Gets confusion matrix
#get_cfmatrix(test_labels, test_img, mod) 
print_misclass(mod,4,5)


# =============================================
# Iterates over different augmented datasets
# =============================================
   
# acc_list=[]
# model= CNN2()
# history = model.fit(train_img, train_labels, epochs=20, 
#                     validation_data=(test_img, test_labels))
# test_loss, test_acc = model.evaluate(test_img,  test_labels, verbose=2)
# acc_list.append(test_acc)

# rot = np.arange(5,11,1)
# for x in rot:
#     # load data frames
#     train_df = pd.read_csv(f"rot_upto{x}.csv")

#     # split training data into features and label
#     train_images = train_df.iloc[:, :-1]
#     train_labels = train_df.iloc[:, -1]

#     #make images into correct shape
#     train_img= np.zeros((5000,16,15))
#     for i in range(5000):
#         train_img[i,:,:] = np.array(train_images.iloc[i,:]).reshape(16,15)
    
#     history = model.fit(train_img, train_labels, epochs=20, 
#                     validation_data=(test_img, test_labels))
#     test_loss, test_acc = model.evaluate(test_img,  test_labels, verbose=2)
#     acc_list.append(test_acc)

# print(acc_list)



