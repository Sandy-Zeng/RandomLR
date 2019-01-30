from keras.optimizers import SGD, RMSprop, Adagrad
import sys
sys.path.append('/home/ouyangzhihao/Backup/Exp/ZYY/RandomLR')
import models.densenet_cifar10 as densenet
from models.LR_resNet import *
from models.vgg import model as vgg
from scheduler.CLR_callback import CyclicLR
from util.DataLoader import DataLoader
from scheduler.lr_scheduler import *
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, LambdaCallback
from keras.callbacks import TensorBoard
import numpy as np
from scheduler.optimizers import SGD_RM
from keras.models import load_model
from models.wide_residual_network import create_wide_residual_network
from models.vgg_bn import model as vgg_bn
from models.vgg_nodropout import model as vgg_nodropout
from scheduler.ladder_scheduler import LadderScheduler
from scheduler.L4_Opt import L4_Mom
import tensorflow as tf

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
    model_name = (sys.argv[12])
    triangle_method = sys.argv[13]
    multFac = int(sys.argv[14])
    random_potion = float(sys.argv[15])
    data_aug = int(sys.argv[16])
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
    triangle_method = 'triangle'
    multFac = 1


work_path = Path('/home/ouyangzhihao/Backup/Exp/ZYY/RandomLR')
work_path = work_path/work_path_name

# work_path = Path('/unsullied/sharefs/ouyangzhihao/DataRoot/Exp/HTB/LID_Research/LR_Random')
max_acc_log_path = work_path/'res.txt'
Te=10

if model_name == 'resnet':
    exp_name = '%s_%d_%d_%s_%s_%s_%.2f_%.4f_%.2f_%d_ResNet_%d_%s' % (dataset_name,epochs,batch_size,optimizer,distribution_method,lr_schedule_method,random_range,linear_init_lr,random_potion,Te,resnet_depth,triangle_method)
if model_name == 'densenet':
    exp_name = '%s_%d_%d_%s_%s_%s_%.2f_%.4f_DenseNet' % (
    dataset_name, epochs, batch_size, optimizer, distribution_method, lr_schedule_method, random_range, linear_init_lr)
if 'vgg' in model_name:
    exp_name = '%s_%d_%d_%s_%s_%s_%.2f_%.4f_%s_%d_%.2f' % (
        dataset_name, epochs, batch_size, optimizer, distribution_method, lr_schedule_method, random_range,
        linear_init_lr,model_name,16,random_potion)
if model_name == 'wrn':
    exp_name = '%s_%d_%d_%s_%s_%s_%.2f_%.4f_WRN' % (
        dataset_name, epochs, batch_size, optimizer, distribution_method, lr_schedule_method, random_range,
        linear_init_lr)

print(exp_name)
if((work_path/'TB_Log'/exp_name).exists()):
    print('Already Finished!')
    exit()

#load data
data_loader = DataLoader(dataset_name=dataset_name)
x_train,y_train,x_test,y_test,num_classes = data_loader.load_data()
input_shape = x_train.shape[1:]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

#define optimizer
init_lr = 0.
base_lr = 0.001
max_lr = 0.
if optimizer=='Adam':
    base_lr = 0.001
    max_lr = 0.006
elif optimizer=='SGD':
    base_lr = 0.1
    max_lr = 0.3
elif optimizer == 'SGD_Mon':
    base_lr = 0.1
    max_lr = 0.3
elif optimizer =='RMSprop':
    max_lr = 0.006
elif optimizer == 'Adagrad':
    max_lr = 0.1

#define scheduler
if lr_schedule_method == 'linear':
    if model_name == 'densenet':
        densenet_scheduler = DenseNetSchedule(epochs=epochs,init_lr=linear_init_lr,distribution_method=distribution_method,random_potion=0.4,random_range=random_range)
        scheduler = densenet_scheduler
    else:
        step_decay = StepDecay(epochs=epochs,init_lr=linear_init_lr,distribution_method=distribution_method,random_potion=random_potion,random_range=random_range)
        scheduler = step_decay
if lr_schedule_method == 'post_random':
    if model_name == 'densenet':
        densenet_scheduler = DenseNetSchedule(epochs=epochs,init_lr=linear_init_lr,distribution_method=distribution_method,random_potion=0.4,random_range=random_range)
        scheduler = densenet_scheduler
    else:
        step_decay = StepDecayPost(epochs=epochs, init_lr=linear_init_lr, distribution_method=distribution_method,
                               random_potion=random_potion, random_range=random_range)
        scheduler = step_decay
if lr_schedule_method == 'batch_random':
    step_decay = BatchRLR(epochs=epochs, init_lr=linear_init_lr, distribution_method=distribution_method,
                           random_potion=random_potion, random_range=random_range)
    scheduler = step_decay
if lr_schedule_method == 'warm_start':
    WS_Scheduler = Warm_Start_Scheduler(init_lr=linear_init_lr,Te=Te,multFac=multFac,distribution_method=distribution_method,random_range=random_range,random_potion=random_potion,epochs=epochs)
    scheduler = WS_Scheduler
if lr_schedule_method == 'clr':
    N = x_train.shape[0]
    iteration = int(N/batch_size)
    print('step_size',8*iteration)
    clr = CyclicLR(base_lr=base_lr, max_lr=max_lr,
                   step_size=8*iteration, mode=triangle_method,
                   distribution_method=distribution_method,
                   calm_down=0,random_range=random_range,
                   epochs=epochs)
    scheduler = clr
    init_lr = base_lr
if lr_schedule_method == 'ladder':
    ladder = LadderScheduler(base_lr=base_lr,max_lr=max_lr,step_size=random_range,calm_down=random_potion,epochs=epochs)
    scheduler = ladder
    init_lr = base_lr
if lr_schedule_method == 'constant':
    scheduler = Constant(init_lr=linear_init_lr,epochs=epochs,distribution_method=distribution_method,random_range=random_range,random_potion=0.4)
if lr_schedule_method == 'exp':
    N = x_train.shape[0]
    iteration = int(N / batch_size)
    print('step_size', iteration)
    global_step = int(iteration * epochs)
    decay_step = 600
    print(global_step)
    scheduler = Exp(init_lr=linear_init_lr,epochs=epochs,decay_step=decay_step,distribution_method=distribution_method,random_range=random_range,random_potion=0.4)
    init_lr = scheduler.lr_schedule()

TB_log_path = work_path/TB_Logs_Path/exp_name

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir,train_data=None):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir)
        update_freq = 'batch'
        if update_freq == 'batch':
            # It is the same as writing as frequently as possible.
            self.update_freq = 1
        else:
            self.update_freq = update_freq
        self.samples_seen = 0
        self.samples_seen_at_last_write = 0
        self.train_data = train_data


    def set_model(self, model):
        self.model = model
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()
        if self.merged is None:
            # 显示学习率的变化
            # self.merged = tf.summary.merge_all()
            self.val_lr_summary = tf.summary.scalar("val_l4_lr", self.model.optimizer.l_rate)
            # self.loss_summary = tf.summary.scalar('min_loss',self.model.optimizer.min_loss)

        self.merged = tf.summary.merge_all()
        self.lr_summary = tf.summary.scalar("l4_lr", self.model.optimizer.l_rate)

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        tlogs = logs or {}

        if self.validation_data:
            val_data = self.validation_data
            tensors = (self.model.inputs +
                       self.model.targets +
                       self.model.sample_weights)

            if self.model.uses_learning_phase:
                tensors += [K.learning_phase()]

            assert len(val_data) == len(tensors)
            val_size = val_data[0].shape[0]
            i = 0
            while i < val_size:
                step = min(self.batch_size, val_size - i)
                if self.model.uses_learning_phase:
                    # do not slice the learning phase
                    batch_val = [x[i:i + step] for x in val_data[:-1]]
                    batch_val.append(val_data[-1])
                else:
                    batch_val = [x[i:i + step] for x in val_data]
                assert len(batch_val) == len(tensors)
                feed_dict = dict(zip(tensors, batch_val))
                # print(self.merged)
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch)
                i += self.batch_size

        if self.update_freq == 'epoch':
            index = epoch
        else:
            index = self.samples_seen
        self._write_logs(logs, index)

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)
        self.writer.flush()

    def on_batch_end(self, batch, logs=None):
        if self.update_freq != 'epoch':
            self.samples_seen += logs['size']
            samples_seen_since = self.samples_seen - self.samples_seen_at_last_write
            if samples_seen_since >= self.update_freq:
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)
                weight_tensor = [1]*batch_size
                weight_tensor = np.asarray(weight_tensor)
                feed = self.train_data + (weight_tensor,)
                feed_dict = dict(zip(tensors, feed))
                summary = self.sess.run(self.lr_summary, feed_dict=feed_dict)
                # lrate = self.sess.run(self.model.optimizer.l_rate, feed_dict=feed_dict)
                # print('lrate',lrate)
                self.writer.add_summary(summary, self.samples_seen)
                self._write_logs(logs, self.samples_seen)
                self.samples_seen_at_last_write = self.samples_seen


# tb_callbacks = LRTensorBoard(log_dir = TB_log_dir)

if lr_schedule_method != 'exp' and lr_schedule_method != 'clr' and lr_schedule_method != 'ladder':
    init_lr = scheduler.lr_schedule(0)
if optimizer=='Adam':
    opt = Adam(lr=init_lr)
elif optimizer=='SGD':
    print(optimizer)
    opt = SGD(lr=init_lr)
elif optimizer == 'SGD_Mon':
    opt = SGD(lr=init_lr,momentum=0.9)
elif optimizer=='SGD_RM':
    opt = SGD_RM(lr=init_lr)
elif optimizer =='RMSprop':
    opt = RMSprop(lr=init_lr)
elif optimizer == 'Adagrad':
    opt = Adagrad(lr=init_lr)
elif optimizer == 'L4':
    opt = L4_Mom()

# Prepare model model saving directory.
# save_dir = os.path.join(os.getcwd(), work_path_name+'/saved_models')
# model_name = '%s.{epoch:03d}.h5' % (exp_name)
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# filepath = os.path.join(save_dir, model_name)
#
# # Prepare callbacks for model saving and for learning rate adjustment.
# checkpoint = ModelCheckpoint(filepath=filepath,
#                              monitor='val_acc',
#                              verbose=1,
#                              save_best_only=True,
#                              period=30)

# saved_models = os.listdir(save_dir)
# for model_file in saved_models:
#     if exp_name in model_file:
#         print('Having Models!')
#         model = load_model(os.path.join(save_dir,model_file))
#         break


#define models
if model_name == 'resnet':
    print(resnet_depth)
    model = resnet_v2(input_shape=input_shape, depth=resnet_depth, num_classes=num_classes)
if model_name == 'densenet':
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
if model_name == 'vgg_bn':
    print(model_name)
    model = vgg_bn(input_shape=input_shape,num_classes=num_classes)
if model_name == 'vgg_nodropout':
    print(model_name)
    model = vgg_nodropout(input_shape=input_shape, num_classes=num_classes)
if model_name == 'wrn':
    model = create_wide_residual_network(input_dim=input_shape,nb_classes=num_classes,N=2,k=8)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy', 'top_k_categorical_accuracy'])
# model.summary()
print("-"*20+exp_name+'-'*20)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
### Train end
last_acc = 0
best_acc = 0
convergence_epoch = 0
def on_epoch_end(epoch, logs):
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

# if optimizer == 'L4':
#     callbacks = [on_epoch_end_callback, scheduler, lr_reducer, LRTensorBoard(log_dir=TB_log_path.__str__(),train_data=)]
# else:
#     callbacks = [on_epoch_end_callback,scheduler,lr_reducer,TensorBoard(log_dir= (TB_log_path.__str__()))]
# callbacks = [on_epoch_end_callback,scheduler,lr_reducer,LRTensorBoard(log_dir = TB_log_path.__str__())]

if data_aug == 1:
    aug = True
else:
    aug = False

if aug == False:
    if optimizer == 'L4':
        callbacks = [on_epoch_end_callback, scheduler, lr_reducer,
                     LRTensorBoard(log_dir=TB_log_path.__str__())]
    else:
        callbacks = [on_epoch_end_callback, scheduler, lr_reducer, TensorBoard(log_dir=(TB_log_path.__str__()))]
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
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
        data_format="channels_last",
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0
    )
    datagen.fit(x_train)
    # Fit the model on the batches generated by datagen.flow().
    generator = datagen.flow(x_train, y_train, batch_size=batch_size)
    if optimizer == 'L4':
        callbacks = [on_epoch_end_callback, scheduler, lr_reducer,
                     LRTensorBoard(log_dir=TB_log_path.__str__(), train_data=generator.__next__())]
    else:
        callbacks = [on_epoch_end_callback, scheduler, lr_reducer, TensorBoard(log_dir=(TB_log_path.__str__()))]
    model.fit_generator(generator=generator,
                        validation_data=(x_test, y_test),
                        steps_per_epoch=len(generator),
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
print(max_acc_log_line, file=open(max_acc_log_path.__str__(), 'a'))
if optimizer == 'L4':
    if os.path.exists('../checkpoint') == False:
        os.mkdir('../checkpoint')
    model.save('../checkpoint/L4_' + str(epochs) + '.h5')
# model.save('./ModelFile/'+exp_name+'.h5')