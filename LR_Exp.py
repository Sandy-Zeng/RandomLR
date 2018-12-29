from models.LR_resNet import *
import time

import math
from keras.optimizers import SGD, RMSprop, Adagrad

from CLR_callback import CyclicLR
from models.LR_resNet import *
from models import densenet

if(len(sys.argv)>11):
    print('Right Parameter')
    dataset_name = (sys.argv[1])
    batch_size = int(sys.argv[2])
    epochs = int(sys.argv[3])
    optimizer = (sys.argv[4])
    lr_schedule_method = (sys.argv[5])
    distribution_method = (sys.argv[6])
    random_range = float(sys.argv[7])
    linear_init_lr = float(sys.argv[8])
    TB_Logs_Path = (sys.argv[9])
    work_path_name = (sys.argv[10]).strip('\r\n')
    resnet_depth = int(sys.argv[11])
    model = (sys.argv[12])
else:
    print('Wrong Params')
    # exit()
    # dataset_name = 'MNIST'
    dataset_name = 'CIFAR10'
    batch_size = 64  # orig paper trained all networks with batch_size=128
    epochs = 20
    optimizer = 'Adam'
    lr_schedule_method = 'clr'
    distribution_method = 'RL'
    random_range = 4
    work_path_name = 'Default'
    linear_init_lr = 1e-3
    TB_Logs_Path = 'TB_Log'
    resnet_depth = 32
    model = 'resnet'


work_path = Path('/home/ouyangzhihao/sss/Exp/ZYY/RandomLR')
work_path = work_path/work_path_name

# work_path = Path('/unsullied/sharefs/ouyangzhihao/DataRoot/Exp/HTB/LID_Research/LR_Random')
max_acc_log_path = work_path/'res.txt'
convergence_epoch = 0

# Training parameters
if model == 'resnet':
    exp_name = '%s_%d_%d_%s_%s_%s_%.2f_%.4f_ResNet_%d' % (dataset_name,epochs,batch_size,optimizer,distribution_method,lr_schedule_method,random_range,linear_init_lr,resnet_depth)
if model == 'densenet':
    exp_name = '%s_%d_%d_%s_%s_%s_%.2f_%.4f_DenseNet' % (
    dataset_name, epochs, batch_size, optimizer, distribution_method, lr_schedule_method, random_range, linear_init_lr)
if((work_path/'TB_Log'/exp_name).exists()):
    print('Already Finished!')
    exit()

##### Train

import keras
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, LambdaCallback
from keras.callbacks import TensorBoard
from keras.datasets import cifar10, cifar100
import numpy as np

# Load the dataset.
if dataset_name=='CIFAR10':
    print(dataset_name)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    num_classes = len(set(y_train.flatten()))
if dataset_name=='CIFAR100':
    print(dataset_name)
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    num_classes = len(set(y_train.flatten()))

print("class num ",num_classes)

# Path(work_path/'Layer_LID_nparray').mkdir(parents=True, exist_ok=True)
input_shape = x_train.shape[1:]

x_train, x_test = preProcessData(x_train,x_test,subtract_pixel_mean=True)

train_num,test_num = x_train.shape[0],x_test.shape[0]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

def lr_schedule(epoch):
    def U(tmp_lr):
        factor = 1e2
        np.random.seed(int(time.time()))
        tmp_lr = np.random.random() * tmp_lr * random_range
        # tmp_lr = np.random.random() * tmp_lr
        return tmp_lr
    def N(tmp_lr,mu=0,sigma=1):
        np.random.seed(int(time.time()))
        tmp_lr_factor = np.random.normal(mu,sigma)
        tmp_lr_factor = abs(tmp_lr_factor)
        tmp_lr *= tmp_lr_factor
        return tmp_lr

    #Learning Rate Schedule
    lr = linear_init_lr
    if epoch >= epochs * 0.9:
        lr *= 0.5e-3
    elif epoch >= epochs * 0.8:
        lr *= 1e-3
    elif epoch >= epochs * 0.6:
        lr *= 1e-2
    elif epoch >= epochs * 0.4:
        lr *= 1e-1

    if distribution_method =='U':
        lr = U(lr)
    elif distribution_method =='Base':
        lr = lr
    print('Learning rate: ', lr)
    return lr

Te = 20
tt = 0
t0 = math.pi/2.0
TeNext = Te
multFactor = 2
cyc = True
def warm_start_lr_schedule(epoch):
    def WRSGN(epoch, tmp_lr):
        global Te, tt, t0, TeNext, multFactor, cyc, epoches
        dt = 2.0 * math.pi / float(2.0 * Te)
        tt = tt + float(dt)
        if tt >= math.pi:
            tt = tt - math.pi
        curT = t0 + tt
        new_lr = tmp_lr * (1.0 + math.sin(curT)) / 2.0  # lr_min = 0, lr_max = lr
        if (epoch + 1 == TeNext):  # time to restart
            tt = 0  # by setting to 0 we set lr to lr_max, see above
            Te = Te * multFactor  # change the period of restarts
            TeNext = TeNext + Te  # note the next restart's epoch
            if TeNext > epochs:
                cyc = False
                return linear_init_lr *1e-3
        return new_lr

    global cyc
    if cyc == True:
        lr = linear_init_lr
        lr = WRSGN(epoch,lr)
    else:
        lr = linear_init_lr * 1e-3
    print('Learning rate: ', lr)
    return lr


init_lr = 0.
base_lr = 0.001
max_lr = 0.
if optimizer=='Adam':
    base_lr = 0.001
    max_lr = 0.006
elif optimizer=='SGD':
    base_lr = 0.1
    max_lr = 0.3
elif optimizer =='RMSprop':
    max_lr = 0.006
elif optimizer == 'Adagrad':
    max_lr = 0.1

if lr_schedule_method == 'clr':
    clr = CyclicLR(base_lr=base_lr, max_lr=max_lr,
                   step_size=5000., mode='triangular2',
                   distribution_method=distribution_method)
    init_lr = base_lr
else:
    init_lr = lr_schedule(0)



#ResNet:
# model = keras.applications.resnet50.ResNet50(input_shape=None, include_top=True, weights=None)
if model == 'resnet':
    model = resnet_v1(input_shape=input_shape, depth=resnet_depth,num_classes = num_classes)
if model == 'densenet':
    model = densenet.DenseNet(classes=num_classes, input_shape=input_shape, depth=40, growth_rate=12,
                              bottleneck=True, reduction=0.5)
if optimizer=='Adam':
    opt = Adam(lr=init_lr)
elif optimizer=='SGD':
    opt = SGD(lr=init_lr)
elif optimizer =='RMSprop':
    opt = RMSprop(lr=init_lr)
elif optimizer == 'Adagrad':
    opt = Adagrad(lr=init_lr)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy', 'top_k_categorical_accuracy'])
# model.summary()
print("-"*20+exp_name+'-'*20)

# Prepare model model saving directory.


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


### Train end
last_acc = 0
best_acc = 0
convergence_epoch = 0
TB_log_path = work_path/TB_Logs_Path/exp_name
def on_epoch_end(epoch, logs):
    # from ipdb import set_trace as tr; tr()
    # print(logs)
    global last_acc
    global best_acc
    if(logs['val_acc'] - last_acc > 0.01):
        global convergence_epoch
        convergence_epoch = epoch
    if(logs['val_acc']>best_acc):
        best_acc = logs['val_acc']
    last_acc = logs['val_acc']
    print('End of epoch')
    # renew_train_dataset()


on_epoch_end_callback = LambdaCallback(on_epoch_end=on_epoch_end)


if lr_schedule_method == 'linear':
    scheduler = LearningRateScheduler(lr_schedule)
if lr_schedule_method == 'clr':
    scheduler = clr
if lr_schedule_method == 'warm_start':
    scheduler = LearningRateScheduler(warm_start_lr_schedule)
callbacks = [on_epoch_end_callback,scheduler,lr_reducer,TensorBoard(log_dir= (TB_log_path.__str__()))]
# Run training, with or without data augmentation.
# Run training, with or without data augmentation.
aug = True
if aug == False:
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
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
        validation_split=0.0
    )

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1,batch_size=batch_size*4)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


### Final result output
final_accuracy = scores[1]
final_loss = scores[0]

print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ("exp_name", 'best_accuracy','final_accuracy', 'final_loss',
                                  'converage_epoch', 'distribution', 'par1', 'par2', 'dataset_name' ))
max_acc_log_line = "%s\t%f\t%f\t%f\t%d\t%s\t%d\t%s" % (exp_name, best_acc,final_accuracy, final_loss, convergence_epoch, distribution_method, random_range, dataset_name)
print(max_acc_log_line)
# print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ("exp_name", 'final_accuracy', 'final_loss',
#                                   'converage_epoch', 'lid_method', 'drop_percent', 'model_name','dataset_name' ),file=open(max_acc_log_path.__str__(), 'a'))
print(max_acc_log_line, file=open(max_acc_log_path.__str__(), 'a'))