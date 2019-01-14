from keras.datasets import cifar10
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import pickle as pk

num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
x_train, x_val = x_train[:512], x_train[40000:]
y_train, y_val = y_train[:512], y_train[40000:]


with open('weights_with_random_retrain.pkl', 'rb') as f:
    weights = pk.load(f)

weights1 = np.asanyarray([each[0] for each in weights])
weights2 = np.asanyarray([each[1] for each in weights])

print(weights1.max(), weights1.min())
print(weights2.max(), weights2.min())

model = load_model('pretrain_model.h5')
layers = model.layers
layer = layers[6]
print(layer.get_weights())

weights_loss_train = []
for weight1 in weights1:
    for weight2 in weights2:
        loss, acc = model.test_on_batch(x_train, y_train)
        print(weight1, weight2, loss)
        weights_loss_train.append([weight1, weight2, loss])


with open('weights_loss_train_with_random.pkl', 'wb') as f:
    pk.dump(weights_loss_train, f)