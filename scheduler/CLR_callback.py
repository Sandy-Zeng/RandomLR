from keras.callbacks import *


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000.,epochs=200, mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle',distribution_method='U',calm_down=20,random_range=1):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.init_base_lr = base_lr
        self.init_max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.distribution_method = distribution_method
        self.calm_down = calm_down
        self.count = calm_down
        self.random_range = random_range
        self.epochs = epochs
        self.israndom = False
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

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
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def U(self,tmp_lr):
        np.random.seed(int(time.time()))
        # tmp_lr = np.random.random() * factor / np.sqrt(factor) * tmp_lr
        tmp_lr = np.random.random() * tmp_lr * self.random_range
        return tmp_lr

    def N(self,tmp_lr, mu=0.2, sigma=0.8):
        np.random.seed(int(time.time()))
        tmp_lr_factor = np.random.normal(mu, sigma)
        tmp_lr_factor = abs(tmp_lr_factor)
        tmp_lr *= tmp_lr_factor
        return tmp_lr

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.count == 0:
            self.trn_iterations += 1
            self.clr_iterations += 1
            self.count = self.calm_down
        else:
            self.count -= 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        if self.distribution_method == 'U':
            if self.israndom == True:
                K.set_value(self.model.optimizer.lr, self.U(self.clr()))
            else:
                K.set_value(self.model.optimizer.lr, self.clr())
        if self.distribution_method == 'N':
            K.set_value(self.model.optimizer.lr, self.N(self.clr()))
        if self.distribution_method == 'Base':
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs=None):
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



