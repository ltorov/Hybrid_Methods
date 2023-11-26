import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import idx2numpy
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall

from src.auxiliary import partitioning
from src.plotting import show_images, show_performance_curve

def leNet(plotting = True):

    # Import the data we want to classify
    our_numbers = pd.read_csv('data/numbers.csv')
    our_numbers = our_numbers.drop(columns=['Unnamed: 0'], axis=1)
    our_numbers = our_numbers[our_numbers["label"]!="X"]

    # Import training and testing data
    train_x = idx2numpy.convert_from_file('data/train_images.idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('data/train_labels.idx1-ubyte')
    test_x = idx2numpy.convert_from_file('data/test_images.idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('data/test_labels.idx1-ubyte')
    class_names = list(set(train_labels))

    if plotting:
        # Visualize training data
        show_images(train_x, class_names, train_labels)

    # Normalize values
    max_pixel_value = 255
    train_images = train_x / max_pixel_value
    test_images = test_x / max_pixel_value

    # Pad images
    train_images = np.pad(train_images, ((0,0),(2,2), (2,2)), 'constant', constant_values = 0)
    test_images = np.pad(test_images, ((0,0),(2,2), (2,2)), 'constant', constant_values =0)

    # Convert categorical values
    train_labels = to_categorical(train_labels, len(class_names))
    test_labels = to_categorical(test_labels, len(class_names))

    # Design Net
    INPUT_SHAPE = (32,32,1)
    FILTER1_SIZE = 6
    FILTER2_SIZE = 16
    FILTER_SHAPE = (5, 5)
    POOL_SHAPE = (2, 2)
    FULLY_CONNECT_NUM = 128
    NUM_CLASSES = len(class_names)
    model = Sequential()
    model.add(Conv2D(FILTER1_SIZE, FILTER_SHAPE,activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(POOL_SHAPE))
    model.add(Conv2D(FILTER2_SIZE, FILTER_SHAPE, activation='relu'))
    model.add(MaxPooling2D(POOL_SHAPE))
    model.add(Flatten())
    model.add(Dense(FULLY_CONNECT_NUM, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # Train model
    BATCH_SIZE = 32
    EPOCHS = 50
    METRICS = metrics=['accuracy',
                    Precision(name='precision'),
                    Recall(name='recall')]
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics = METRICS)
    training_history = model.fit(train_images, train_labels,
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(test_images, test_labels))
    
    # Visualize training through the performance curve
    if plotting: 
        show_performance_curve(training_history, 'accuracy', 'Accuracy')
        show_performance_curve(training_history, 'precision', 'Precision')

    if plotting:
        # Create confusion matrix
        test_predictions = model.predict(test_images)

        test_predicted_labels = np.argmax(test_predictions, axis=1)

        test_true_labels = np.argmax(test_labels, axis=1)

        cm = confusion_matrix(test_true_labels, test_predicted_labels)
        sumas = np.sum(cm,axis=1)
        cm2 = np.zeros(cm.shape)
        for i in range(len(cm)):
            cm2[i] = np.round(cm[i]/sumas[i],2)

        cmd = ConfusionMatrixDisplay(confusion_matrix=cm2)

        cmd.plot(include_values=True, cmap='Greys', ax=None, xticks_rotation='horizontal')
        plt.show()

    # Prepare for validation
    validation = our_numbers.iloc[:,:-1].values
    validation = validation.reshape(714,28, 28)
    validation_images =  (1-validation) / 1
    validation_images = np.pad(validation_images, ((0,0),(2,2), (2,2)), 'constant', constant_values =0)

    # Validate
    predictions = model.predict(validation_images)
    predicted_classes = predictions.argmax(axis=-1)

    if plotting: 
        show_images(validation_images,class_names,predicted_classes)
    
    if plotting:
        # Create confusion matrix
        cm = confusion_matrix([int(i) for i in our_numbers.iloc[:,-1].values], predicted_classes)

        sumas = np.sum(cm,axis=1)
        cm2 = np.zeros(cm.shape)
        for i in range(len(cm)):
            cm2[i] = np.round(cm[i]/sumas[i],2)

        cmd = ConfusionMatrixDisplay(confusion_matrix=cm2)

        cmd.plot(include_values=True, cmap='Greys', ax=None, xticks_rotation='horizontal')
        plt.show()