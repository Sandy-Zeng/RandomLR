from keras.callbacks import *

class LadderScheduler(Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., epochs=200, mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle', distribution_method='U', calm_down=20, random_range=1):
        super(LadderScheduler,self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.init_base_lr = base_lr
        self.init_max_lr = max_lr
        self.step_size = step_size
        self.epochs = epochs
        self.calm_down = calm_down
        self.count = self.calm_down
        self.iteration = 0

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.iteration = 0.

    def step(self):
        piece = (self.max_lr-self.base_lr)/float(self.step_size)
        if self.iteration < self.step_size:
            added_lr = piece * self.iteration
            lr = self.base_lr + added_lr
        else:
            added_lr = piece * (self.iteration-self.step_size)
            lr = self.max_lr - added_lr
        return lr

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        if self.count == 0:
            self.count = self.calm_down
            self.iteration = (self.iteration+1)%(self.step_size*2)
            if self.iteration == 0:
                self.max_lr = self.max_lr/2
        else:
            self.count = self.count - 1
        lr = self.step()
        K.set_value(self.model.optimizer.lr, lr)
        print('Learning Rate:',lr)
        print('Iteration:',self.iteration)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        if epoch == self.epochs * 0.9:
            self._reset(new_base_lr=self.init_base_lr*0.5e-3,new_max_lr=self.init_max_lr*0.5e-3)
        elif epoch == self.epochs * 0.8:
            self._reset(new_base_lr=self.init_base_lr * 1e-3, new_max_lr=self.init_max_lr * 1e-3)
        elif epoch == self.epochs * 0.6:
            self._reset(new_base_lr=self.init_base_lr * 1e-2, new_max_lr=self.init_max_lr * 1e-2)
        elif epoch == self.epochs * 0.4:
            self.israndom = True
            self._reset(new_base_lr=self.init_base_lr * 1e-1, new_max_lr=self.init_max_lr * 1e-1)
        print(self.max_lr)
        print(self.base_lr)






