#Filename:	mnist.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Rab 07 Jul 2021 07:15:04 

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
from sklearn.metrics import accuracy_score

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == "__main__":

    num_classes = 10
    input_shape = (28, 28, 1)
    train = preprocessing.image_dataset_from_directory("../data/MNIST_V1/train", labels = "inferred", label_mode = "categorical", color_mode = "grayscale",
            image_size = (28, 28), batch_size = 128)

    test = preprocessing.image_dataset_from_directory("../data/MNIST_V1/test", labels = "inferred", label_mode = "categorical", color_mode = "grayscale",
            image_size = (28, 28), batch_size = 128)
    
    datagen = preprocessing.image.ImageDataGenerator(rescale = 1./255)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    batch_size = 128
    epochs = 100
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(train, epochs=epochs)

    # evaluate
    prediction = np.array([])
    labels = np.array([])
    for data, label in test:
        prediction = np.concatenate([prediction, np.argmax(model.predict(data), -1)])
        labels = np.concatenate([labels, np.argmax(label.numpy(), axis = -1)])
    
    print(accuracy_score(prediction, labels))

    model.save("mnist_cnn.h5")
