from keras.datasets import cifar10

from models.LR_resNet import *


class DataLoader():

    def __init__(self,dataset_name):
        self.dataset_name = dataset_name

    def load_data(self):
        if self.dataset_name == 'CIFAR10':
            print(self.dataset_name)
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            num_classes = len(set(y_train.flatten()))
        if self.dataset_name == 'CIFAR100':
            print(self.dataset_name)
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
            num_classes = len(set(y_train.flatten()))
        num_classes = len(set(y_train.flatten()))

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        input_shape = x_train.shape[1:]

        x_train, x_test = self.preProcessData(x_train, x_test, subtract_pixel_mean=True)
        datagen = self.data_augment(x_train=x_train)
        return x_train,y_train,x_test,y_test,datagen,num_classes

    def preProcessData(self,x_train, x_test, subtract_pixel_mean=True):
        # Normalize data.
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # If subtract pixel mean is enabled
        if subtract_pixel_mean:
            x_train_mean = np.mean(x_train, axis=0)
            x_train -= x_train_mean
            x_test -= x_train_mean
        return x_train, x_test

    def data_augment(self,x_train):
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=10,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        # datagen.fit(x_train)
        return datagen
