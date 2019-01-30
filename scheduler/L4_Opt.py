from keras.optimizers import Optimizer
from keras import backend as K
if K.backend() == 'tensorflow':
    import tensorflow as tf
from keras.legacy import interfaces


def n_inner_product(list_of_tensors1, list_of_tensors2):
    return tf.add_n([tf.reduce_sum(t1*t2) for t1, t2 in zip(list_of_tensors1, list_of_tensors2)])


def time_factor(time_step):
    """ Routine used for bias correction in exponential moving averages, as in (Kingma, Ba, 2015) """
    global_step = 1 + tf.train.get_or_create_global_step()
    decay = 1.0 - 1.0 / time_step
    return 1.0 - K.exp((K.cast(global_step, tf.float32)) * K.log(decay))

class MomentumTransform(object):
    """
    Class implementing momentum transform of the gradient (here in the form of exponential moving average)
    """
    def __init__(self, time_momentum=10.0):
        self.time_momentum = time_momentum
        self.EMAgrad = tf.train.ExponentialMovingAverage(decay=1.0-1.0/self.time_momentum)

    def momgrad(self, grads):
        shadow_op_gr = self.EMAgrad.apply(grads)
        with tf.control_dependencies([shadow_op_gr]):
            correction_term = time_factor(self.time_momentum)
            new_grads = [self.EMAgrad.average(grad) / correction_term for grad in grads]
            return [tf.identity(grad) for grad in new_grads]


class L4_Mom(Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False,fraction=0.15, minloss_factor=0.9, init_factor=0.75,
                 minloss_forget_time=1000.0, epsilon=1e-12,
                 gradient_estimator='momentum', gradient_params=None,
                 direction_estimator='adam', direction_params=None,**kwargs):
        super(L4_Mom, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.l_rate = K.variable(0,name='effective_lf')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
            self.min_loss = K.variable(0,name='min_loss')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.fraction = fraction
        self.minloss_factor = minloss_factor
        self.minloss_increase_rate = 1.0 + 1.0 / minloss_forget_time
        self.epsilon = epsilon
        self.init_factor = init_factor

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        # if self.iterations == 0:
        #     print('aaaaa')
        #     self.min_loss = self.init_factor * loss
        # else:
        #     self.min_loss = K.min(self.min_loss,loss)
        ml_newval = tf.cond(tf.equal(self.iterations, 0), lambda: self.init_factor * loss,
                            lambda: tf.minimum(self.min_loss, loss))
        self.loss = loss
        self.updates = [K.update_add(self.iterations, 1)]

        min_loss_to_use = self.minloss_factor * self.min_loss
        # momtransform = MomentumTransform()
        # new_grads = momtransform.momgrad(grads)
        self.l_rate = self.fraction * (loss - min_loss_to_use) / (
            n_inner_product(grads,grads) + self.epsilon)

        self.min_loss = self.minloss_increase_rate * self.min_loss

        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - self.l_rate * g  # velocity
            self.updates.append(K.update(m, v))
            new_p = p + v
            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(L4_Mom, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))