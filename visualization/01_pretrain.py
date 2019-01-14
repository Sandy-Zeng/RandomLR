import keras
from keras.layers import Dense, Flatten, Input
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from logger import *
import pickle as pk
from lr_scheduler import *

save_logs(sys.argv[0])


class SaveWeights(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        weights_and_biases = self.model.layers[6].get_weights()
        weight1 = weights_and_biases[0][0][0]
        weight2 = weights_and_biases[0][0][1]
        print('wegiht1: ', weight1, 'weight2: ', weight2)
        self.weights.append([weight1, weight2])

    # def on_batch_begin(self, batch, logs=None):
    #     layer = self.model.layers[6]
    #     weights_and_biases = self.model.layers[6].get_weights()
    #     weight1 = weights_and_biases[0][0][0]
    #     weight2 = weights_and_biases[0][0][1]
    #     layer.set_weights([np.array([[weight1, weight2]]), np.array([0.0, 0.0])])
    #
    # def on_batch_end(self, batch, logs=None):
    #     layer = self.model.layers[6]
    #     weights_and_biases = self.model.layers[6].get_weights()
    #     weight1 = weights_and_biases[0][0][0]
    #     weight2 = weights_and_biases[0][0][1]
    #     layer.set_weights([np.array([[weight1, weight2]]), np.array([0.0, 0.0])])

    def on_train_end(self, logs=None):
        self.weights = np.asanyarray(self.weights)
        with open('weights.pkl', 'wb') as f:
            pk.dump(self.weights, f)


batch_size = 32
epochs = 200
num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
x_train, x_val = x_train[:40000], x_train[40000:]
y_train, y_val = y_train[:40000], y_train[40000:]

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255)
x_test = x_test / 255.0
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size)

conv_base = VGG16(weights='imagenet', include_top=False)
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    layer.trainable = set_trainable

input_tensor = Input(shape=x_train[0].shape)
x = conv_base(input_tensor)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
x = Dense(8, activation='relu')(x)
x = Dense(1, activation='relu')(x)
x = Dense(2, activation='relu')(x)
x = Dense(10, activation='relu')(x)
output_tensor = Dense(10, activation='softmax')(x)
model = Model(input_tensor, output_tensor)
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(train_generator, epochs=epochs, validation_data=val_generator, verbose=2,
                    steps_per_epoch=math.ceil(x_train.shape[0] // batch_size),
                    validation_steps=math.ceil(x_val.shape[0]) // batch_size,
                    callbacks=[SaveWeights()])

model.save('pretrain_model.h5')
