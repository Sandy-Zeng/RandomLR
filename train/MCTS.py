import sys
import time
from random import choice
import sys
sys.path.append('/home/ouyangzhihao/Backup/Exp/ZYY/RandomLR')

import math
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, LambdaCallback
from keras.callbacks import TensorBoard
from keras.optimizers import *
from sklearn.model_selection import train_test_split

from models.LR_resNet import resnet_v2
from util.DataLoader import DataLoader
from models.LR_resNet import *


class Node(object):
    def __init__(self,init_model_weight,depth,parent,learning_rate):
        self.Q = 0.
        self.N = 1
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
        if self.parent != None:
            print('Parent Depth:',self.parent.depth)
        print('LR:',self.learning_rate)
        print('Q:',self.Q)
        print('N:',self.N)
        print('UST:',self.ust_score)

    def is_fully_expanded(self):
        if len(self.children) == 0:
            return False
        for child in self.children:
            # print('child N:',child.N)
            if child.N == 1:
                return False
        return True

    def update_ust(self):
        for child in self.children:
            child.ust_score = child.Q / (child.N) + 0.1 * math.sqrt(math.log(self.N) / (child.N))

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
                best_ust = child.ust_score
        print('-----pick best ust-----')
        best_child.print_node()
        return best_child

    def pick_unvisited(self):
        for child in self.children:
            if child.N == 1:
                return child
        return None

    def is_root(self):
        if self.parent == None:
            return True
        return False

    def hightest_visit(self):
        best_lr = self.learning_rate
        max_N = 1
        best_child = self
        for child in self.children:
            if child.N > max_N:
                max_N = child.N
                best_lr = child.learning_rate
                best_child = child
            if max_N==1 and child.model_weight!=None:
                max_N = child.N
                best_lr = child.learning_rate
                best_child = child
        return best_lr,best_child

    def gen_children(self):
        epoch = self.depth
        lr = decay_lr(epoch)
        for i in range(random_num):
            temp_lr = U(lr, random_range)
            child = Node(init_model_weight=None, depth=epoch + 1, parent=self, learning_rate=temp_lr)
            self.children.append(child)



class MCTS(object):
    def __init__(self,model_weight,init_depth,total_depth,power,random_range,random_num,eval_x,eval_y,mini_epoch):
        self.power = power
        self.random_num = random_num
        self.best_lrs = []
        self.eval_x = eval_x
        self.eval_y = eval_y
        self.total_depth = total_depth
        self.mini_epoch = mini_epoch
        self.root = None
        self.init_depth = init_depth
        self.generate_hole_tree(init_depth=init_depth,model_weight=model_weight,tree_depth=total_depth,random_range=random_range,random_num=random_num)

    def generate_hole_tree(self,init_depth,model_weight,tree_depth,random_range,random_num):
        self.root = Node(init_model_weight=model_weight, depth=init_depth, parent=None, learning_rate=init_lr)
        layer_cache = []
        layer_cache.append(self.root)
        for epoch in range(tree_depth):
            lr = decay_lr(init_depth+1+epoch)
            # lr = init_lr
            prod_cache = []
            print('Depth:', init_depth+epoch)
            print('Node Num:', len(layer_cache))
            print('Base_lr:',lr)
            for inter_node in layer_cache:
                for i in range(random_num):
                    # randomly generate the learning rate
                    temp_lr = U(lr, random_range)
                    print(temp_lr)
                    child = Node(init_model_weight=None, depth=epoch+init_depth+1, parent=inter_node, learning_rate=temp_lr)
                    inter_node.children.append(child)
                    prod_cache.append(child)
            layer_cache = prod_cache

    def resources_left(self):
        self.power = self.power - 1
        print('Power Left:',self.power)
        if self.power >= 0:
            return True
        else:
            return False

    def monte_carlo_tree_search(self):
        while self.resources_left():
            leaf = self.traverse(self.root) # leaf = unvisited node
            simulation_result = self.rollout(leaf)
            self.backpropagate(leaf,simulation_result)
        return self.best_child(self.root)

    def traverse(self,node):
        while node.is_fully_expanded():
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
        if node.depth< self.init_depth + self.total_depth:
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
            validation_data=(x_val, y_val),
            epochs=self.mini_epoch, verbose=1, workers=4,
        )
        return local_model.get_weights()


    def rollout_policy(self,node):

        #generate children
        if node.depth <= self.total_depth and len(node.children)==0:
            node = node.gen_children()

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
            scores = model.evaluate(x_val,y_val, verbose=1, batch_size=batch_size * 4)
            # scores = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size * 4)
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
            print('-------best child----------')
            node.print_node()
        return node.model_weight

def iterate_gen_tree(total_depth,sub_tree_depth,model_weight):
    sub_tree_num = int(total_depth/(sub_tree_depth))
    best_lrs = []
    depth = -1
    temp_weight = model_weight
    for i in range(sub_tree_num):
        MCTS_Tree = MCTS(model_weight=temp_weight,init_depth=depth,total_depth=sub_tree_depth,
                         power=power,random_range=random_range,
                         random_num=random_num, eval_x=x_train_mini,
                         eval_y=y_train_mini, mini_epoch=mini_epoch)
        temp_weight = MCTS_Tree.monte_carlo_tree_search()
        print(MCTS_Tree.best_lrs)
        best_lrs.extend(MCTS_Tree.best_lrs)
        depth += sub_tree_depth

    residul_depth = total_depth - (sub_tree_depth)*sub_tree_num
    if residul_depth>0:
        MCTS_Tree = MCTS(model_weight=temp_weight, init_depth=depth, total_depth=residul_depth,
                         power=power, random_range=random_range,
                         random_num=random_num, eval_x=x_train_mini,
                         eval_y=y_train_mini, mini_epoch=mini_epoch)
        model_weight = MCTS_Tree.monte_carlo_tree_search()
        print(MCTS_Tree.best_lrs)
        best_lrs.extend(MCTS_Tree.best_lrs)
        depth += residul_depth

    print('Depth:',residul_depth)
    print(best_lrs)

    return best_lrs

def new_model(global_model_weight,optimizer,input_shape,num_classes,lr):
    # model = resnet_v1(input_shape=input_shape, depth=5 * 6 + 2, num_classes=num_classes)
    global model
    model.set_weights(global_model_weight)
    K.set_value(model.optimizer.lr, lr)
    return model

def U(tmp_lr,random_range):
    # np.random.seed(int(time.time()))
    # factor = 0.17
    # rand = (random_range * np.random.random()+1)
    # print(rand)
    # tmp_lr = tmp_lr ** (factor * rand)
    tmp_lr = tmp_lr * (random_range * np.random.random())
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
    model = resnet_v2(input_shape=input_shape, depth=20, num_classes=num_classes)
    if optimizer == 'Adam':
        opt = Adam(lr=init_lr)
    elif optimizer == 'SGD':
        opt = SGD(lr=init_lr,momentum=0.9)
    elif optimizer == 'RMSprop':
        opt = RMSprop(lr=init_lr)
    elif optimizer == 'Adagrad':
        opt = Adagrad(lr=init_lr)

    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    TB_log_path = work_path/work_path_name/'TB_Logs'/exp_name
    on_epoch_end_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    callbacks = [on_epoch_end_callback,lr_scheduler, lr_reducer, TensorBoard(log_dir=(TB_log_path.__str__()))]
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
    t1 = time.time()
    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_test, y_test),
        epochs=epochs, verbose=1, workers=4,
        callbacks=callbacks
    )
    scores = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size * 4)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    t2 = time.time()
    evaluate_time = t2-t1
    percentage = MCTS_TIME/evaluate_time
    print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t" % ("exp_name", 'best_accuracy', 'final_accuracy', 'final_loss',
                                                  'converage_epoch','MCTS time','evaluate time','percentage'))
    max_acc_log_line = "%s\t%f\t%f\t%f\t%d\t%.2f\t%.2f\t%.4f" % (
    exp_name, best_acc, scores[1], scores[0], convergence_epoch,MCTS_TIME,evaluate_time,percentage)
    max_acc_log_path = work_path/work_path_name/'res.txt'
    print(max_acc_log_line)
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
    sub_tree_depth = int(sys.argv[8])
    power = int(sys.argv[9])
    mini_epoch = int(sys.argv[10])
    work_path_name = str(sys.argv[11])

    from pathlib import *
    work_path = Path('/home/ouyangzhihao/Backup/Exp/ZYY/RandomLR')

    total_depth = int(epochs/mini_epoch)  #20
    # mini_epoch = int(epochs / total_depth) #10

    exp_name = '%s_%d_%d_%s_%d_%.4f_%d_%d_%d_%d_%d_MCTS' % (
        dataset_name, epochs, batch_size, optimizer, random_range, init_lr, random_num, power,mini_epoch,sub_tree_depth,0.1
    )
    if ((work_path / work_path_name / 'TB_Logs' / exp_name).exists()):
        print('Already Finished!')
        exit()

    data_loader = DataLoader(dataset_name=dataset_name)
    x_train, y_train, x_test, y_test,num_classes = data_loader.load_data()
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
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0
    )
    datagen.fit(x_train)

    train_sample_num = x_train.shape[0]
    mini_train_sample_num = train_sample_num * 0.1
    print('Mini Train Num:',mini_train_sample_num)
    _,x_train_MCTS,_,y_train_MCTS = train_test_split(x_train,y_train,test_size=0.2)
    x_train_mini, x_val, y_train_mini, y_val = train_test_split(x_train_MCTS,y_train_MCTS,test_size=0.5)
    datagen.fit(x_train_mini)
    assert x_train_mini.shape[0]==mini_train_sample_num
    input_shape = x_train.shape[1:]

    model = resnet_v2(input_shape=input_shape, depth=20, num_classes=num_classes)

    # root = generate_hole_tree(model_weight=model.get_weights(),mini_epochs=3,init_lr=init_lr,random_range=random_range,random_num=random_num)
    # test_gen_hole_tree(root)
    if optimizer == 'Adam':
        opt = Adam(lr=init_lr)
    elif optimizer == 'SGD':
        opt = SGD(lr=init_lr,momentum=0.9)
    elif optimizer == 'RMSprop':
        opt = RMSprop(lr=init_lr)
    elif optimizer == 'Adagrad':
        opt = Adagrad(lr=init_lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    print("-" * 20 + exp_name + '-' * 20)
    # print('------------Generate mini tree---------------------')
    start = time.time()
    best_lrs = iterate_gen_tree(total_depth=total_depth, sub_tree_depth=sub_tree_depth, model_weight=model.get_weights())
    end = time.time()
    MCTS_TIME = end - start
    print('MCTS cost:',end-start)

    # print('------------MCTS search--------------')
    # MCTS_Tree = MCTS(power=power,total_depth=total_depth,random_num=random_num,eval_x=x_train_mini,eval_y=y_train_mini,mini_epoch=mini_epoch)
    # MCTS_Tree.monte_carlo_tree_search(root=root)
    # best_lrs = MCTS_Tree.best_lrs
    # print(MCTS_Tree.best_lrs)

    # best_lrs = [0.00077299085,0.0014,0.002240,0.002559,0.001476,0.0015318,0.0041795,0.0032571,0.000361128,0.00370699,0.001599,4.22e-5,2.764e-5,1.44036e-5,4.10257e-5,2.8634756e-6,1.621935e-6,4.20071916e-6,2.7859968e-7,1.51925e-6]
    print('-----------Evaluate Learning Rates---------------')
    evaluate(init_lr=init_lr,exp_name=exp_name)










