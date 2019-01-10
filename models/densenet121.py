import keras.applications.densenet as densenet
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input

def DenseNet(classes=10,input_shape=(32,32,3)):
    input_tensor = Input(shape=input_shape)
    base_model = densenet.DenseNet121(include_top=False, weights=None, input_tensor=input_tensor,input_shape=input_shape,
                                 pooling='avg', classes=classes)
    # 添加全局平均池化层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # 添加一个全连接层
    x = Dense(1024, activation='relu')(x)

    # 添加一个分类器，假设我们有200个类
    predictions = Dense(classes, activation='softmax')(x)

    # 构建我们需要训练的完整模型
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

