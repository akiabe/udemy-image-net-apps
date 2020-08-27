import os
import shutil
import random

def split_data(SOURCE, TRAIN, VALID, SPLIT_SIZE):
    """
    :param SOURCE: source directory containing the files
    :param TRAIN: train directory that a portion of the files will be copied to
    :param VALID: valid directory that a portion of the files will be copied to
    :param SPLIT_SIZE: split size to determine the portion
    :return:
    """
    all_files = []

    # pick up file_name from source directory and joint path to source directory
    for file_name in os.listdir(SOURCE):
        file_path = SOURCE + file_name

        # if file size in file_path is bigger than 0, append file_name to list
        if os.path.getsize(file_path) > 0:
            all_files.append(file_name)
        else:
            print(file_name + "is zero length, so ignoring.")

    n_files = len(all_files)
    split_point = int(n_files * SPLIT_SIZE)

    # random.sample(list, len(list)) shuffles a list
    shuffled = random.sample(all_files, n_files)
    train_set = shuffled[:split_point]
    valid_set = shuffled[split_point:]

    # copyfile(source, destination) copies a file from source to destination
    for file_name in train_set:
        shutil.copyfile(SOURCE + file_name, TRAIN + file_name)

    for file_name in valid_set:
        shutil.copyfile(SOURCE + file_name, VALID + file_name)


if __name__ == "__main__":
    # define directory for train and valid sets
    try:
        os.mkdir('../input/train')
        os.mkdir('../input/valid')
        os.mkdir('../input/train/boar')
        os.mkdir('../input/train/crow')
        os.mkdir('../input/train/monkey')
        os.mkdir('../input/valid/boar')
        os.mkdir('../input/valid/crow')
        os.mkdir('../input/valid/monkey')
    except OSError:
        pass

    BOAR_SOURCE_DIR = "../input/boar/"
    CROW_SOURCE_DIR = "../input/crow/"
    MONKEY_SOURCE_DIR = "../input/monkey/"
    TRAIN_BOAR_DIR = "../input/train/boar/"
    TRAIN_CROW_DIR = "../input/train/crow/"
    TRAIN_MONKEY_DIR = "../input/train/monkey/"
    VALID_BOAR_DIR = "../input/valid/boar/"
    VALID_CROW_DIR = "../input/valid/crow/"
    VALID_MONKEY_DIR = "../input/valid/monkey/"

    split_size = .9
    split_data(BOAR_SOURCE_DIR, TRAIN_BOAR_DIR, VALID_BOAR_DIR, split_size)
    split_data(CROW_SOURCE_DIR, TRAIN_CROW_DIR, VALID_CROW_DIR, split_size)
    split_data(MONKEY_SOURCE_DIR, TRAIN_MONKEY_DIR, VALID_MONKEY_DIR, split_size)

    print(len(os.listdir("../input/train/boar/")))
    print(len(os.listdir("../input/train/crow/")))
    print(len(os.listdir("../input/train/monkey/")))
    print(len(os.listdir("../input/valid/boar/")))
    print(len(os.listdir("../input/valid/crow/")))
    print(len(os.listdir("../input/valid/monkey/")))