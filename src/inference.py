import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image

from PIL import Image
import os, sys
import numpy as np

if __name__ == "__main__":
    # build the model
    model = tf.keras.models.Sequential([
        # Input shape is the desired size of the image 150x150 with 3 bytes color
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # set optimizer
    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(lr=1e-4),
        metrics=['accuracy']
    )

    # load trained model
    model = tf.keras.models.load_model("../models/animal-net.h5")

    uploaded = Image.open(sys.argv[1])

    img = uploaded.convert('RGB')
    img = img.resize((150, 150))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model.predict(images, batch_size=10)
    print(classes)

    #print(type(images))
    #print(images.shape)



