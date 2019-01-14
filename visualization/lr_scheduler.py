import numpy as np
import time
import math
from keras.callbacks import *

def U(tmp_lr,random_range):
    np.random.seed(int(time.time()))
    tmp_lr = np.random.random() * tmp_lr * random_range
    # tmp_lr = tmp_lr + tmp_lr * np.random.random()
    return tmp_lr

def UA(tmp_lr,random_range):
    np.random.seed(int(time.time()))
    tmp_lr = tmp_lr + tmp_lr * np.random.random() * random_range
    return tmp_lr


def N(tmp_lr, mu=4, sigma=1):
    np.random.seed(int(time.time()))
    tmp_lr_factor = np.random.normal(mu, sigma)
    tmp_lr_factor = abs(tmp_lr_factor) * tmp_lr
    tmp_lr = tmp_lr + tmp_lr_factor
    return tmp_lr

class StepDecay(Callback):

    def __init__(self,epochs=200,init_lr=1e-3,distribution_method='N',random_potion=0.3,random_range=10):
        super(StepDecay, self).__init__()
        self.epochs = epochs
        self.linear_init_lr = init_lr
        self.distribution_method = distribution_method
        self.random_potion = random_potion
        self.random_range = random_range
        self.count_down = 19
        self.count = 0
        self.random_lr = init_lr
        self.last_lr = init_lr
        self.beta = 0.5

    def lr_schedule(self,epoch):
        #Learning Rate Schedule
        lr = self.linear_init_lr
        left = 0
        right = self.epochs * 0.4
        if epoch > self.epochs * 0.9:
            lr *= 0.5e-3
            left = self.epochs * 0.9
            right = self.epochs
        elif epoch > self.epochs * 0.8:
            lr *= 1e-3
            left = self.epochs * 0.8
            right = self.epochs * 0.9
        elif epoch > self.epochs * 0.6:
            lr *= 1e-2
            left = self.epochs * 0.6
            right = self.epochs * 0.8
        elif epoch > self.epochs * 0.4:
            lr *= 1e-1
            left = self.epochs * 0.4
            right = self.epochs * 0.6

        if epoch == self.epochs * 0.9+1:
            self.last_lr = self.linear_init_lr * 0.5e-3
        elif epoch == self.epochs * 0.8+1:
            self.last_lr = self.linear_init_lr * 1e-3
        elif epoch == self.epochs * 0.6+1:
            self.last_lr = self.linear_init_lr * 1e-2
        elif epoch == self.epochs * 0.4+1:
            self.last_lr = self.linear_init_lr * 1e-1

        bounder = left + int((right - left) * self.random_potion)
        if epoch < bounder:
            print('Bounder:', bounder)
            if self.distribution_method == 'U':
                # if (epoch - left) < ((right - left)*(self.random_potion/2)):
                #     adaptive_range = (epoch-left)/float((right - left) * (self.random_potion)/2) * self.random_range + 0.1
                #     lr = U(lr,adaptive_range)
                # else:
                #     lr = U(lr,self.random_range+0.1)
                # adaptive_range = (right - epoch) / float(
                #     (right - left)) * self.random_range + 0.1
                # lr = U(lr, adaptive_range)
                lr = U(lr, self.random_range)
                # lr = (lr + self.last_lr)/2
                lr = self.beta * self.last_lr + (1-self.beta)*lr
                self.last_lr = lr
            if self.distribution_method == 'UC':
                if self.count == 0:
                    lr = U(lr,self.random_range)
                    self.random_lr = lr
                    self.count = self.count_down
                else:
                    lr = self.random_lr
                    self.count -= 1
            if self.distribution_method == 'N':
                lr = N(tmp_lr=lr,mu=self.random_range)
            elif self.distribution_method == 'Base':
                lr = lr
        print('Learning rate: ', lr)
        return lr

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        lr = self.lr_schedule(epoch=epoch)
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

class StepDecayPost(Callback):

    def __init__(self, epochs=200, init_lr=1e-3, distribution_method='N', random_portion=0.3, random_range=10):
        super(StepDecayPost, self).__init__()
        self.epochs = epochs
        self.linear_init_lr = init_lr
        self.distribution_method = distribution_method
        self.random_portion = random_portion
        self.random_range = random_range
        self.count_down = 19
        self.count = 0
        self.random_lr = init_lr

    def lr_schedule(self,epoch):
        #Learning Rate Schedule
        lr = self.linear_init_lr
        left = 0
        right = self.epochs * 0.4
        if epoch > self.epochs * 0.9:
            lr *= 0.5e-3
            left = self.epochs * 0.9
            right = self.epochs
        elif epoch > self.epochs * 0.8:
            lr *= 1e-3
            left = self.epochs * 0.8
            right = self.epochs * 0.9
        elif epoch > self.epochs * 0.6:
            lr *= 1e-2
            left = self.epochs * 0.6
            right = self.epochs * 0.8
        elif epoch > self.epochs * 0.4:
            lr *= 1e-1
            left = self.epochs * 0.4
            right = self.epochs * 0.6

        bounder = left + int((right - left) * self.random_portion)
        if epoch < bounder and epoch>self.epochs*0.4:
            print('Bounder:', bounder)
            if self.distribution_method == 'U':
                lr = U(lr, self.random_range)
            if self.distribution_method == 'UA':
                lr = UA(lr,self.random_range)
            if self.distribution_method == 'UC':
                if self.count == 0:
                    lr = U(lr,self.random_range)
                    self.random_lr = lr
                    self.count = self.count_down
                else:
                    lr = self.random_lr
                    self.count -= 1
            if self.distribution_method == 'N':
                lr = N(lr)
            elif self.distribution_method == 'Base':
                lr = lr
        print('Learning rate: ', lr)
        return lr

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.lr_schedule(epoch=epoch)
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


class BatchRLR(Callback):

    def __init__(self,epochs=200,init_lr=1e-3,distribution_method='N',random_potion=0.3,random_range=10):
        super(BatchRLR, self).__init__()
        self.epochs = epochs
        self.linear_init_lr = init_lr
        self.distribution_method = distribution_method
        self.random_potion = random_potion
        self.random_range = random_range
        self.count_down = 19
        self.count = 0
        self.last_lr = init_lr
        self.beta = 0.7
        self.base_lr = init_lr

    def lr_schedule(self,batch):
        #Learning Rate Schedule
        lr = self.base_lr
        if self.distribution_method == 'U':
            lr = U(lr, self.random_range)
            lr = self.beta * self.last_lr + (1-self.beta) * lr
        if self.distribution_method == 'N':
            lr = N(lr,random_range=self.random_range)
        elif self.distribution_method == 'Base':
            lr = lr
        return lr

    def on_batch_begin(self, batch, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))
        lr = self.lr_schedule(batch=batch)
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch > self.epochs * 0.9:
            self.base_lr *= 0.5e-3
        elif epoch > self.epochs * 0.8:
            self.base_lr *= 1e-3
        elif epoch > self.epochs * 0.6:
            self.base_lr *= 1e-2
        elif epoch > self.epochs * 0.4:
            self.base_lr *= 1e-1

        if epoch == self.epochs * 0.9 + 1:
            self.last_lr = self.linear_init_lr * 0.5e-3
        elif epoch == self.epochs * 0.8 + 1:
            self.last_lr = self.linear_init_lr * 1e-3
        elif epoch == self.epochs * 0.6 + 1:
            self.last_lr = self.linear_init_lr * 1e-2
        elif epoch == self.epochs * 0.4 + 1:
            self.last_lr = self.linear_init_lr * 1e-1

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

class Constant(Callback):

    def __init__(self,epochs=200,init_lr=1e-3,distribution_method='N',random_potion=0.3,random_range=10):
        super(Constant, self).__init__()
        self.epochs = epochs
        self.linear_init_lr = init_lr
        self.distribution_method = distribution_method
        self.random_potion = random_potion
        self.random_range = random_range

    def lr_schedule(self,epoch):
        #Learning Rate Schedule
        lr = self.linear_init_lr
        left = 0
        right = self.epochs

        bounder = left + int((right - left) * self.random_potion)
        if epoch < bounder:
            print('Bounder:', bounder)
            if self.distribution_method == 'U':
                lr = U(lr,self.random_range)
            if self.distribution_method == 'N':
                lr = N(lr,mu=self.random_range)
            elif self.distribution_method == 'Base':
                lr = lr
        print('Learning rate: ', lr)
        return lr

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        lr = self.lr_schedule(epoch=epoch)
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

class DenseNetSchedule(Callback):
    def __init__(self,epochs=300,init_lr=1e-3,distribution_method='N',random_range=10,random_potion=0.3):
        super(DenseNetSchedule,self).__init__()
        self.epochs = epochs
        self.linear_init_lr = init_lr
        self.distribution_method = distribution_method
        self.random_range = random_range
        self.random_potion = random_potion

    def lr_schedule(self,epoch):

        # Learning Rate Schedule
        lr = self.linear_init_lr
        left = 0
        right = self.epochs * 0.5
        if epoch >= self.epochs * 0.75:
            lr *= 1e-2
            left = self.epochs * 0.75
            right = self.epochs
        elif epoch >= self.epochs * 0.5:
            lr *= 1e-1
            left = self.epochs * 0.5
            right = self.epochs * 0.75

        bounder = left + int((right - left) * self.random_potion)
        if epoch < bounder and epoch>= self.epochs*0.5:
            print('Bounder:', bounder)
            if self.distribution_method == 'U':
                lr = U(lr, self.random_range)
            if self.distribution_method == 'N':
                lr = N(lr, mu=self.random_range)
            elif self.distribution_method == 'Base':
                lr = lr
        print('Learning rate: ', lr)
        return lr

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # lr = float(K.get_value(self.model.optimizer.lr))
        lr = self.lr_schedule(epoch=epoch)
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

class Warm_Start_Scheduler(Callback):
    def __init__(self,init_lr=1e-3,Te=10,multFac=2,distribution_method='N',random_range=10,random_potion=0.5,epochs=200):
        super(Warm_Start_Scheduler,self).__init__()
        self.Te = Te
        self.tt = 0
        self.t0 = math.pi / 2.0
        self.TeNext = Te
        self.multFactor = multFac
        self.init_lr = init_lr
        self.distribution_method = distribution_method
        self.random_range = random_range
        self.random_potion = random_potion
        self.epochs = epochs
        self.iscycle = True
        self.last_lr = init_lr

    def lr_schedule(self,epoch):
        def WRSGN(epoch, tmp_lr):
            dt = 2.0 * math.pi / float(2.0 * self.Te)
            self.tt = self.tt + float(dt)
            if self.tt >= math.pi:
                self.tt = self.tt - math.pi
            curT = self.t0 + self.tt
            new_lr = tmp_lr * (1.0 + math.sin(curT)) / 2.0  # lr_min = 0, lr_max = lr
            if (epoch + 1 == self.TeNext):  # time to restart
                self.tt = 0  # by setting to 0 we set lr to lr_max, see above
                self.Te = self.Te * self.multFactor  # change the period of restarts
                self.TeNext = self.TeNext + self.Te  # note the next restart's epoch
                if self.TeNext > self.epochs:
                    self.iscycle = False
                    self.last_lr = new_lr
            return new_lr

        lr = self.init_lr
        if self.iscycle:
            lr = WRSGN(epoch, lr)
        else:
            lr = self.last_lr
        if epoch < self.epochs * self.random_potion and epoch>80 and epoch<130:
            if self.distribution_method == 'U':
                lr = U(lr, self.random_range)
            if self.distribution_method == 'N':
                lr = N(lr, mu=self.random_range)
            elif self.distribution_method == 'Base':
                lr = lr
        print('Learning rate: ', lr)
        return lr

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        lr = self.lr_schedule(epoch=epoch)
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

class Exp(Callback):
    def __init__(self,epochs=200,init_lr=1e-3,decay_rate=0.96,decay_step=1000,distribution_method='N',random_potion=0.3,random_range=10):
        super(Exp,self).__init__()
        self.epochs = epochs
        self.linear_init_lr = init_lr
        self.distribution_method = distribution_method
        self.random_potion = random_potion
        self.random_range = random_range
        self.decay_rate = decay_rate
        self.global_step = 0.
        self.decay_step = decay_step
        self.history = {}
        self.israndom = False

    def lr_schedule(self):
        lr = self.linear_init_lr
        lr = lr * math.pow(self.decay_rate,math.floor(self.global_step/ self.decay_step))
        if self.israndom == True:
            if self.distribution_method == 'U':
                lr = U(lr, self.random_range)
            if self.distribution_method == 'N':
                lr = N(lr, mu=self.random_range)
            elif self.distribution_method == 'Base':
                lr = lr
        # print('Learning rate: ', lr)
        return lr

    def on_train_begin(self, logs={}):
        logs = logs or {}
        print(self.global_step)
        if self.global_step == 0:
            print(self.linear_init_lr)
            K.set_value(self.model.optimizer.lr, self.linear_init_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.lr_schedule())

    def on_batch_end(self, epoch, logs=None):
        # lr = float(K.get_value(self.model.optimizer.lr))
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.global_step)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.global_step = self.global_step + 1
        lr = self.lr_schedule()
        K.set_value(self.model.optimizer.lr, lr)

    # def on_epoch_end(self, epoch, logs=None):
    #     logs = logs or {}
    #     logs['lr'] = K.get_value(self.model.optimizer.lr)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        logs['lr'] = lr
        print('Learning Rate:',lr)
        if epoch > 80 and epoch<130:
            self.israndom = True
        else:
            self.israndom = False

class RetinaSchedule(Callback):
    def __init__(self,epochs=150,init_lr=1e-1,distribution_method='N',random_range=10):
        super(RetinaSchedule,self).__init__()
        self.epochs = epochs
        self.linear_init_lr = init_lr
        self.distribution_method = distribution_method
        self.random_range = random_range

    def lr_schedule(self,epoch):

        # Learning Rate Schedule
        lr = self.linear_init_lr
        if epoch > 140:
            lr *= 1e-2
        elif epoch > 120:
            lr *= 1e-1

        if epoch>120:
            if self.distribution_method == 'U':
                lr = U(lr, self.random_range)
            if self.distribution_method == 'N':
                lr = N(lr, mu=self.random_range)
            elif self.distribution_method == 'Base':
                lr = lr
        print('Learning rate: ', lr)
        return lr

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # lr = float(K.get_value(self.model.optimizer.lr))
        lr = self.lr_schedule(epoch=epoch)
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)







