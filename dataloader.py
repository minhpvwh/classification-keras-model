import numpy as np
import keras
import cv2
import glob
import config as cfg
# import train



class DataGenerator(keras.utils.Sequence):

    """
        Data Generator: Customize Dataloader Keras for Image Classify
    """

    def __init__(self, imgs_path, batch_size, n_channels=1, shuffle=True):

        """ Initialization"""

        self.batch_size = batch_size
        self.imgs_path = imgs_path
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):

        """ Denotes the number of batches per epoch """

        return int(np.floor(len(self.imgs_path)) / self.batch_size)

    def __getitem__(self, index):

        """generate one batch of data_agu"""
        # try:
        # generate index of the batch data_agu
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # find list of IDs
        list_paths = [self.imgs_path[k] for k in indexes]
        X, y = self.__data_generation(list_paths)

        return X, y
        # except Exception as e:
        #     pass

    def on_epoch_end(self):
        """Updates indexes after each epoch """
        self.indexes = np.arange(len(self.imgs_path))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_path_imgs):
        """ generate data_agu containing batch_size samples"""

        images = []
        label = []
        #gen data_agu
        for path in list_path_imgs:
            # store sample
            img = cv2.imread(path)
            cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = cv2.resize(cvt_img, (300, 100))
            image = image.astype('float32')/255
            images.append(image)

            # TODO re-implement converter key label dict in config.py 
            key = path.split("/")[6]
            label_id = cfg.label_dict[key]
            label.append(label_id)
                

        X_train = np.array(images)
        Y_train = keras.utils.to_categorical(label, num_classes=cfg.NUM_CLASSES)

        return X_train, [Y_train]


def get_dataset(dataset, ratio=0.8):
    """
        split dataset for train and evaluate
    """
    paths =  glob.glob(dataset + "/*/*.jpg")
    train = int(len(paths)*80/100)

    return paths[:train], paths[train:]


if __name__ == '__main__':
    # Test
    params = {'batch_size': 16,
              'n_channels': 1,
              'shuffle': True}
    # check exists
    list_paths_train, list_paths_test = train.get_dataset(cfg.dataset, cfg.ratio)
    print(list_paths_train)
    DataGenerator(list_paths_train, **params)