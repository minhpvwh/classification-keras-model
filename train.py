import keras
import glob
import random
import argparse
from collections import Counter
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense
from dataloader import DataGenerator
import cv2
import numpy as np
import pickle
import os
import config as cfg
import tensorflow as tf
from keras import backend as K
from collections import Counter
from tqdm import tqdm



# config GPU usage
K.tensorflow_backend._get_available_gpus()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def check_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_dataset(dataset, ratio=0.8):
    """
        split dataset for train and evaluate
    """
    paths =  glob.glob(dataset + "/*/*.jpg")
    train = int(len(paths)*ratio)

    return paths[:train], paths[train:]

#Model
def main():

    vgg_model = keras.applications.VGG16(weights='imagenet',
                                include_top=False,
                                input_shape=(100, 300, 3))
    output_tensor = vgg_model.get_layer('block5_pool').output
    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(256, activation = 'relu')(output_tensor)
    output_tensor = Dense(128, activation = 'relu')(output_tensor)
    output_tensor = Dropout(0.5)(output_tensor)
    output_tensor = Dense(cfg.NUM_CLASSES, activation = 'softmax', name="classify_embeding")(output_tensor)

    from keras.models import Model
    model = Model(input = vgg_model.input, output = output_tensor)

    for layer in model.layers[:-6]:
        layer.trainable = False

    # opt = keras.optimizers.Adam(lr=0.001)
    model.compile(loss="categorical_crossentropy", optimizer='Adam', metrics=['accuracy'])
    print(model.summary())


    # Parameters
    params = {'batch_size': cfg.BATCH_SIZE,
              'n_channels': 1,
              'shuffle': True}
    # check exists
    check_exists("checkpoints")
    check_exists("logs")
    check_exists("model")
    list_paths_train, list_paths_test = get_dataset(cfg.dataset, cfg.ratio)
    train_generator = DataGenerator(list_paths_train, **params)
    test_generator = DataGenerator(list_paths_test, **params)

    checkpoint = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.001,
                                  min_lr=0)
    # fit
    # print("train generator: ", train_generator)
    model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        use_multiprocessing=False,
                        workers=4,
                        verbose=2,
                        callbacks=[reduce_lr, checkpoint],
                        epochs=cfg.EPOCHS)

    # save model
    model_json = model.to_json()
    with open("model/classify.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model/classify.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    main()
