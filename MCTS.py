import math
import time
import numpy as np
from random import choice
from LR_resNet import resnet_v1
from keras.datasets import cifar10,mnist,cifar100
from DataLoader import DataLoader
import sys
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, LambdaCallback
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

class Node(object):
    def __init__(self,init_model_weight,depth,parent,learning_rate):
        self.Q = 0.
        self.N = 0
        self.mimic_score = 0.
        self.model_weight = init_model_weight
        self.depth = depth
        self.parent = parent
        self.children = []
        self.ust_score = 0
        self.learning_rate = learning_rate

    def print_node(self):
        print('Pick node')
        print('Depth:',self.depth)
        print('Parent Depth:',self.parent.depth)
        print('LR:',self.learning_rate)
        print('Q:',self.Q)
        print('N:',self.N)
        print('UST:',self.ust_score)

    def fully_expanded(self):
        for child in self.children:
            print('child N:',child.N)
            if child.N == 0:
                return False
        return True

    def update_ust(self):
        for child in self.children:
            child.ust_score = child.Q / (child.N+1e-4) + 0.2 * math.sqrt(math.log(self.N) / (child.N+1e-4))

    def update_status(self,result):
        self.Q = self.Q + result
        self.N = self.N + 1
        self.update_ust()
        print('----update status-----')
        self.print_node()


    def best_ust(self):
        best_child = self.children[0]
        best_ust = 0
        for child in self.children:
            if child.ust_score > best_ust:
                best_child = child
        return best_child

    def pick_unvisited(self):
        for child in self.children:
            if child.N == 0:
                return child
        return None

    def is_root(self):
        if self.parent == None:
            return True
        return False

    def hightest_visit(self):
        best_lr = self.learning_rate
        max_N = 0
        best_child = None
        for child in self.children:
            if child.N > max_N:
                max_N = child.N
                best_lr = child.learning_rate
                best_child = child
        return best_lr,best_child



class MCTS(object):
    def __init__(self,total_depth,power,random_num,eval_x,eval_y,mini_epoch):
        self.power = power
        self.random_num = random_num
        self.best_lrs = []
        self.eval_x = eval_x
        self.eval_y = eval_y
        self.total_depth = total_depth
        self.mini_epoch = mini_epoch

    def resources_left(self):
        self.power = self.power - 1
        if self.power > 0:
            return True
        else:
            return False

    def traverse_root(self,root):
        for i in range(random_num):
            child = root.children[i]
            local_model = new_model(root.model_weight, optimizer, input_shape, num_classes, lr=child.learning_rate)
            local_model.fit_generator(
                datagen.flow(self.eval_x, self.eval_y, batch_size=batch_size),
                validation_data=(x_test, y_test),
                epochs=self.mini_epoch, verbose=1, workers=4,
            )
            child.model_weight = local_model.get_weights()


    def monte_carlo_tree_search(self,root):
        while self.resources_left():
            leaf = self.traverse(root) # leaf = unvisited node
            simulation_result = self.rollout(leaf)
            self.backpropagate(leaf,simulation_result)
        return self.best_child(root)

    def traverse(self,node):
        while node.fully_expanded():
            node = node.best_ust()
        child = node.pick_unvisited()
        # in case no children are present /node is terminal
        if child == None:
            return node
        else:
            return child

    def non_terminal(self,node):
        if node == None:
            return False
        if node.depth<self.total_depth:
            return True
        else:
            return False

    def minic(self,parent_model_weight,lr):
        print('LR:',lr)
        local_model = new_model(parent_model_weight, optimizer, input_shape, num_classes, lr=lr)
        # local_model.fit(self.eval_x, self.eval_y,
        #           batch_size=batch_size,
        #           epochs=self.mini_epoch,
        #           validation_data=(x_test, y_test),
        #           shuffle=True)
        local_model.fit_generator(
            datagen.flow(self.eval_x, self.eval_y, batch_size=batch_size),
            validation_data=(x_test, y_test),
            epochs=self.mini_epoch, verbose=1, workers=4,
        )
        return local_model.get_weights()


    def rollout_policy(self,node):

        #generate children
        if node.depth <= self.total_depth and len(node.children)==0:
            node = gen_children(node)

        #randomly pick a children
        list = range(self.random_num)
        index = choice(list)
        child = node.children[index]

        if child.model_weight == None:
            print('Depth:',child.depth)
            # print('LR：',child.learning_rate)
            local_model_weight = self.minic(parent_model_weight=node.model_weight,lr=child.learning_rate)
            child.model_weight = local_model_weight

        return child

    def result(self,node):
        if  node.model_weight == None:
            local_model_weight =self.minic(parent_model_weight=node.parent.model_weight,lr=node.learning_rate)
            node.model_weight = local_model_weight
        if node.mimic_score != 0:
            print(node.mimic_score)
            return node.mimic_score
        else:
            model.set_weights(node.model_weight)
            scores = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size * 4)
            print('mimic acc:', scores[1])
            print('mimic loss:', scores[0])
            node.mimic_score = scores[1]
            # node.Q = scores[1]
            return scores[1]

    def rollout(self,node):
        while self.non_terminal(node):
            print('--------- roll out ---------')
            node.print_node()
            if node.model_weight == None:
                node.model_weight = self.minic(parent_model_weight=node.parent.model_weight,lr=node.learning_rate)
            node = self.rollout_policy(node)
            break
        return self.result(node)

    def backpropagate(self,node,simulate_result):
        node.update_status(simulate_result)
        if node.is_root()== True:
            return
        self.backpropagate(node.parent,simulate_result)

    def best_child(self,node):
        # pick child with highest number of visits
        while self.non_terminal(node) == True:
            lr,node= node.hightest_visit()
            self.best_lrs.append(lr)
        return

def new_model(global_model_weight,optimizer,input_shape,num_classes,lr):
    # model = resnet_v1(input_shape=input_shape, depth=5 * 6 + 2, num_classes=num_classes)
    model.set_weights(global_model_weight)
    K.set_value(model.optimizer.lr, lr)
    return model

def U(tmp_lr,random_range):
    # np.random.seed(int(time.time()))
    tmp_lr = np.random.random() * tmp_lr * random_range + tmp_lr/random_range
    return tmp_lr

def decay_lr(epoch):
    lr = init_lr
    if epoch >= total_depth * 0.9:
        lr *= 0.5e-3
    elif epoch >= total_depth * 0.8:
        lr *= 1e-3
    elif epoch >= total_depth * 0.6:
        lr *= 1e-2
    elif epoch >= total_depth * 0.4:
        lr *= 1e-1
    return lr


def gen_children(node):
    epoch = node.depth
    lr = decay_lr(epoch)
    for i in range(random_num):
        temp_lr = U(lr, random_range)
        child = Node(init_model_weight=None, depth=epoch+1, parent=node, learning_rate=temp_lr)
        node.children.append(child)
    return node




def generate_hole_tree(model_weight,mini_epochs,init_lr,random_range,random_num):
    root = Node(init_model_weight=model_weight,depth=-1,parent=None,learning_rate=init_lr)
    layer_cache = []
    layer_cache.append(root)
    for epoch in range(mini_epochs):
        lr = decay_lr(epoch)
        prod_cache = []
        print('Epoch:',epoch)
        print('Node Num:',len(layer_cache))
        for inter_node in layer_cache:
            for i in range(random_num):
                #randomly generate the learning rate
                temp_lr = U(lr,random_range)
                child = Node(init_model_weight=None,depth=epoch,parent=inter_node,learning_rate=temp_lr)
                inter_node.children.append(child)
                prod_cache.append(child)
        layer_cache = prod_cache
    return root


def lr_schedule(epoch):
    index = int(epoch/mini_epoch)
    if index >= len(best_lrs):
        lr = best_lrs[-1]
    else:
        lr = best_lrs[index]
    print('Learning rate: ', lr)
    return lr

last_acc = 0.
best_acc = 0.
def on_epoch_end(epoch, logs):
    global last_acc,best_acc
    if(logs['val_acc'] - last_acc > 0.01):
        global convergence_epoch
        convergence_epoch = epoch
    if(logs['val_acc']>best_acc):
        best_acc = logs['val_acc']
    last_acc = logs['val_acc']
    print('End of epoch')

def evaluate(init_lr,exp_name):
    model = resnet_v1(input_shape=input_shape, depth=5 * 6 + 2, num_classes=num_classes)
    if optimizer == 'Adam':
        opt = Adam(lr=init_lr)
    elif optimizer == 'SGD':
        opt = SGD(lr=init_lr)
    elif optimizer == 'RMSprop':
        opt = RMSprop(lr=init_lr)
    elif optimizer == 'Adagrad':
        opt = Adagrad(lr=init_lr)

    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    TB_log_path = work_path/'TB_Logs'/exp_name
    callbacks = [on_epoch_end,lr_scheduler, lr_reducer, TensorBoard(log_dir=(TB_log_path.__str__()))]
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_test, y_test),
        epochs=epochs, verbose=1, workers=4,
        callbacks= callbacks
    )
    scores = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size * 4)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    print("%s\t%s\t%s\t%s\t%s\t" % ("exp_name", 'best_accuracy', 'final_accuracy', 'final_loss',
                                                  'converage_epoch'))
    max_acc_log_line = "%s\t%f\t%f\t%f\t%d\t" % (
    exp_name, best_acc, scores[1], scores[0], convergence_epoch)
    max_acc_log_path = work_path/'res.txt'
    print(max_acc_log_line, file=open(max_acc_log_path.__str__(), 'a'))

def test_gen_hole_tree(root):
    layer_cache = []
    layer_cache.append(root)
    for epoch in range(epochs):
        print('-------------------------')
        print('Epoch:',epoch)
        cache = []
        for node in layer_cache:
            print('depth:',node.depth)
            print('lr:',node.learning_rate)
            for i in range(random_num):
                cache.append(node.children[i])
        layer_cache = cache






if __name__ == '__main__':
    dataset_name = (sys.argv[1])
    batch_size = int(sys.argv[2])
    epochs = int(sys.argv[3])
    optimizer = (sys.argv[4])
    random_range = int(sys.argv[5])
    init_lr = float(sys.argv[6])
    random_num = int(sys.argv[7])
    power = int(sys.argv[8])
    work_path = (sys.argv[9])

    exp_name = '%s_%d_%d_%s_%d_%.2f_%d_%d_MCTS' % (
         dataset_name, epochs, batch_size, optimizer, random_range,init_lr,random_num,power
    )

    total_depth = int(epochs * 0.1)  #20
    mini_epoch = int(epochs / total_depth) #10

    data_loader = DataLoader(dataset_name=dataset_name)
    x_train, y_train, x_test, y_test, datagen,num_classes = data_loader.load_data()
    datagen.fit(x_train)

    train_sample_num = x_train.shape[0]
    mini_train_sample_num = train_sample_num * 0.1
    print('Mini Train Num:',mini_train_sample_num)
    _,x_train_mini,_,y_train_mini = train_test_split(x_train,y_train,test_size=0.1)
    datagen.fit(x_train_mini)
    assert x_train_mini.shape[0]==mini_train_sample_num
    input_shape = x_train.shape[1:]

    model = resnet_v1(input_shape=input_shape, depth=5 * 6 + 2, num_classes=num_classes)
    print('------------Generate mini tree---------------------')
    root = generate_hole_tree(model_weight=model.get_weights(),mini_epochs=3,init_lr=init_lr,random_range=random_range,random_num=random_num)
    # test_gen_hole_tree(root)
    if optimizer == 'Adam':
        opt = Adam(lr=init_lr)
    elif optimizer == 'SGD':
        opt = SGD(lr=init_lr)
    elif optimizer == 'RMSprop':
        opt = RMSprop(lr=init_lr)
    elif optimizer == 'Adagrad':
        opt = Adagrad(lr=init_lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])


    print('------------MCTS search--------------')
    MCTS_Tree = MCTS(power=power,total_depth=total_depth,random_num=random_num,eval_x=x_train_mini,eval_y=y_train_mini,mini_epoch=mini_epoch)
    MCTS_Tree.monte_carlo_tree_search(root=root)
    best_lrs = MCTS_Tree.best_lrs
    print(MCTS_Tree.best_lrs)

    print('-----------Evaluate Learning Rates---------------')
    evaluate(init_lr=init_lr,exp_name=exp_name)










