from models.LR_resNet import *
import time
import time
import models.densenet_cifar10 as densenet
from models.vgg import model as vgg

from keras.optimizers import SGD, RMSprop, Adagrad

from models.LR_resNet import *

if(len(sys.argv)>8):
    print('Right Parameter')
    dataset_name = (sys.argv[1])
    batch_size = int(sys.argv[2])
    epochs = int(sys.argv[3])
    optimizer = (sys.argv[4])
    distribution_method = (sys.argv[5])
    lr_schedule_method = (sys.argv[6])
    random_range = float(sys.argv[7])
    peak_delay = float(sys.argv[8])
    linear_init_lr = float(sys.argv[9])
    work_path_name = (sys.argv[10]).strip('\r\n')
    depth = int(sys.argv[11])
    model_name = sys.argv[12]
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
    dis_parameter1 = 0.2
    dis_parameter2 = 0.8
    work_path_name = 'Default'
    linear_init_lr = 1e-3
    model_name = 'resnet'


work_path = Path('/home/ouyangzhihao/sss/Exp/ZYY/RandomLR')
work_path = work_path/work_path_name

# work_path = Path('/unsullied/sharefs/ouyangzhihao/DataRoot/Exp/HTB/LID_Research/LR_Random')
max_acc_log_path = work_path/'res.txt'
convergence_epoch = 0

# Training parameters
# exp_name = '%s_%d_%d_%s_%s_%s_%.2f_%.2f_%.4f_ResNet%d' % (dataset_name,epochs,batch_size,optimizer,distribution_method,lr_schedule_method,random_range,peak_delay,linear_init_lr,depth)
if model_name == 'resnet':
    exp_name = '%s_%d_%d_%s_%s_%s_%.2f_%.4f_%d_ResNet_%d' % (dataset_name,epochs,batch_size,optimizer,distribution_method,lr_schedule_method,random_range,linear_init_lr,peak_delay,depth)
if model_name == 'densenet':
    exp_name = '%s_%d_%d_%s_%s_%s_%.2f_%.4f_DenseNet' % (
    dataset_name, epochs, batch_size, optimizer, distribution_method, lr_schedule_method, random_range, linear_init_lr)
if model_name == 'vgg':
    exp_name = '%s_%d_%d_%s_%s_%s_%.2f_%.4f_VGG' % (
        dataset_name, epochs, batch_size, optimizer, distribution_method, lr_schedule_method, random_range,
        linear_init_lr)

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
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    num_classes = len(set(y_train.flatten()))

print("class num ",num_classes)

# Path(work_path/'Layer_LID_nparray').mkdir(parents=True, exist_ok=True)
input_shape = x_train.shape[1:]

x_train, x_test = preProcessData(x_train,x_test,dataset_name,subtract_pixel_mean=True)

train_num,test_num = x_train.shape[0],x_test.shape[0]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

acc_cache = []
models_cache = []
pointer = -3
peak_value = 0
peak_epoch = 0
peak_delay_temp = peak_delay
decay_period = 0
calm_count = 0
calm_down_lr = 0


def peak_detect(acc,cur_model,epoch):
    global peak_value,peak_delay_temp,model,decay_period,pointer,peak_epoch,peak_delay
    global calm_count,calm_down_lr
    acc_cache.append(acc)
    models_cache.append(cur_model)
    pointer = pointer + 1
    if pointer >=0:
        if acc_cache[pointer+1]>acc_cache[pointer] and acc_cache[pointer+1]>acc_cache[pointer+2]:
            #detect peak
            temp_peak = acc_cache[pointer+1]
            if temp_peak > 0.8:
                print('acc peak', pointer + 1)
                print('peak value',peak_value)
                print('peak delay',peak_delay_temp)
                if temp_peak>peak_value+0.003:
                    peak_value = temp_peak
                    peak_epoch = pointer + 1
                    peak_delay_temp = peak_delay
                    calm_down_lr = K.get_value(model.optimizer.lr)
                elif peak_delay_temp>0:
                    peak_delay_temp = peak_delay_temp- 1
                elif peak_delay_temp == 0:
                    model = models_cache[peak_epoch]
                    peak_delay_temp = peak_delay
                    peak_value = 0
                    decay_period = min(4,decay_period)
                    calm_count = lr_epoch[decay_period] - epoch


def calm_down(acc,cur_model):
    global pointer,acc_cache,models_cache,is_random,calm_count,decay_period
    acc_cache.append(acc)
    models_cache.append(cur_model)
    pointer = pointer + 1
    calm_count = calm_count - 1
    if calm_count == 0:
        decay_period = min(decay_period + 1, 4)


def range_decay(epoch):
    return (epochs - epoch)/float(epochs) * random_range



lr_decay = [linear_init_lr,linear_init_lr*1e-1,linear_init_lr*1e-2,linear_init_lr*1e-3,linear_init_lr*0.5e-3]
lr_epoch = [epochs*0.4,epochs*0.6,epochs*0.8,epochs*0.9,epochs]


def U(tmp_lr):
    np.random.seed(int(time.time()))
    # range = range_decay(epoch)
    tmp_lr = np.random.random() * tmp_lr * random_range
    return tmp_lr

def lr_schedule(epoch):

    #Learning Rate Schedule
    global decay_period,calm_count
    lr = linear_init_lr
    if epoch > epochs * 0.9:
        lr *= 0.5e-3
    elif epoch > epochs * 0.8:
        lr *= 1e-3
    elif epoch > epochs * 0.6:
        lr *= 1e-2
    elif epoch > epochs * 0.4:
        lr *= 1e-1
    early_lr = lr_decay[decay_period]
    # lr = early_lr
    if early_lr < lr:
        lr = early_lr
    print('Base LR:',lr)
    if calm_count == 0:
        if distribution_method =='U':
            lr = U(lr)
        elif distribution_method =='Base':
            lr = lr
    print('Learning rate: ', lr)
    return lr

def densenet_lr_schedule(epoch):

    #Learning Rate Schedule
    lr = linear_init_lr
    if epoch >= epochs * 0.75:
        lr *= 1e-2
    elif epoch >= epochs * 0.5:
        lr *= 1e-1

    if calm_count == 0:
        if distribution_method =='U':
            lr = U(lr)
        elif distribution_method =='Base':
            lr = lr
    print('Learning rate: ', lr)
    return lr


init_lr = lr_schedule(0)
#ResNet:
if model_name == 'resnet':
    model = resnet_v2(input_shape=input_shape, depth=depth, num_classes=num_classes)
if model_name == 'densenet':
    print(model_name)
    model = densenet.DenseNet(nb_classes=num_classes,
                              img_dim=input_shape,
                              depth=40,
                              nb_dense_block=3,
                              growth_rate=12,
                              nb_filter=16,
                              dropout_rate=0,
                              weight_decay=1e-4)
if model_name == 'vgg':
    model = vgg(input_shape=input_shape,num_classes=num_classes)
# model = keras.applications.resnet50.ResNet50(input_shape=None, include_top=True, weights=None)
# model = resnet_v1(input_shape=input_shape, depth=depth,num_classes = num_classes)
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

# model.compile(loss=mycrossentropy,
#               optimizer=opt,
#               metrics=['accuracy', 'top_k_categorical_accuracy'])

# model.summary()
print("-"*20+exp_name+'-'*20)

# Prepare model model saving directory.
if model_name == 'densenet':
    scheduler = LearningRateScheduler(densenet_lr_schedule)
else:
    scheduler = LearningRateScheduler(lr_schedule)
# lr_scheduler = LearningRateScheduler(lr_schedule)


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
if 'noAug' in work_path_name:
    aug = False
    exp_name = exp_name + 'noAug'
else:
    aug = True
TB_log_path = work_path/'TB_Logs'/exp_name


def on_epoch_end(epoch, logs):
    global last_acc,best_acc,model,calm_count
    if calm_count == 0:
        peak_detect(logs['acc'],model,epoch)
    else:
        calm_down(logs['acc'],model)
    if(logs['val_acc'] - last_acc > 0.01):
        global convergence_epoch
        convergence_epoch = epoch
    if(logs['val_acc']>best_acc):
        best_acc = logs['val_acc']
    last_acc = logs['val_acc']
    print('End of epoch')

def random_crop_image(image,pad=4):
    height, width = image.shape[:2]
    zero_border_side = np.zeros((height,pad,3))
    zero_border_top = np.zeros((pad,width+2*pad,3))
    image_pad = np.concatenate((zero_border_side,image),axis=1)
    image_pad = np.concatenate((image_pad,zero_border_side),axis=1)
    image_pad = np.concatenate((zero_border_top,image_pad),axis=0)
    image_pad = np.concatenate((image_pad,zero_border_top),axis=0)
    # print(image_pad.shape)
    # print(image_pad[:,:,0])
    pad_hight,pad_width = image_pad.shape[:2]
    dy = np.random.randint(0,pad_hight-height)
    dx = np.random.randint(0,pad_width-width)
    image_crop = image_pad[dy:dy+height,dx:dx+width,:]
    # print(image_crop.shape)
    # assert False
    return image_crop

on_epoch_end_callback = LambdaCallback(on_epoch_end=on_epoch_end)


# scheduler = lr_scheduler

callbacks = [on_epoch_end_callback,scheduler,lr_reducer,TensorBoard(log_dir= (TB_log_path.__str__()))]
# Run training, with or without data augmentation.
# Run training, with or without data augmentation.
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
        validation_split=0.0)

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
max_acc_log_line = "%s\t%f\t%f\t%f\t%d\t%s\t%d\t%s\t%s" % (exp_name, best_acc,final_accuracy, final_loss, convergence_epoch, distribution_method, random_range, peak_delay, dataset_name)
print(max_acc_log_line)
# print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ("exp_name", 'final_accuracy', 'final_loss',
#                                   'converage_epoch', 'lid_method', 'drop_percent', 'model_name','dataset_name' ),file=open(max_acc_log_path.__str__(), 'a'))
print(max_acc_log_line, file=open(max_acc_log_path.__str__(), 'a'))