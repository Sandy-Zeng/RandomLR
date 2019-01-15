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


with open('weights.pkl', 'rb') as f:
    weights_no_random = pk.load(f)

weights1 = np.asanyarray([each[0] for each in weights_no_random])
weights2 = np.asanyarray([each[1] for each in weights_no_random])

print(weights1.max(), weights1.min())
print(weights2.max(), weights2.min())

model = load_model('pretrain_model.h5')
layers = model.layers
layer = layers[6]
print(layer.get_weights())

weights_loss = []
for weight1 in np.linspace(weights1.min()-0.5, weights1.max()+0.5, 100):
    for weight2 in np.linspace(weights2.min()-0.5, weights2.max()+0.5, 100):
        layer.set_weights([np.array([[weight1,weight2]]), np.array([0,0])])
        loss, acc = model.test_on_batch(x_train, y_train)
        print(weight1, weight2, loss)
        weights_loss.append([weight1, weight2, loss])


with open('weights_loss_all.pkl', 'wb') as f:
    pk.dump(weights_loss, f)