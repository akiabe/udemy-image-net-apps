import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

import image_generator

if __name__ == "__main__":
    # set the image generator
    TRAIN_DIR = "../input/train/"
    VALID_DIR = "../input/valid/"

    train_generator, valid_generator = image_generator.data_generator(TRAIN_DIR, VALID_DIR)

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

    # train the model
    history = model.fit(
        train_generator,
        epochs=25,
        steps_per_epoch=108,  # 1080 images = batch_size * steps
        validation_data=valid_generator,
        verbose=1,
        validation_steps=12  # 120 images = batch_size * steps
    )

    model.save("../models/animal-net.h5")

    