def data_generator(TRAIN_DIR, VALID_DIR):
    """
    :param TRAIN_DIR: train set directory
    :param VALID_DIR: valid_set directory
    :return train_datagen: train generator which flow from train set
    :return valid_datagen: valid generator which flow from valid set
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # set train set generator
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # set valid set generator
    valid_datagen = ImageDataGenerator(rescale=1./255)

    # flow train image to batch
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        batch_size=10,
        class_mode='categorical',
        target_size=(150, 150)
    )

    # flow valid image to batch
    valid_generator = valid_datagen.flow_from_directory(
        VALID_DIR,
        batch_size=10,
        class_mode='categorical',
        target_size=(150, 150)
    )

    return train_generator, valid_generator

