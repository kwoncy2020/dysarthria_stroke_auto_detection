import os
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

BATCH_SIZE = 10
ACTIVATION_FN = 'elu'
filters = 32

class MHAttention(tf.keras.layers.Layer):
    def __init__(self, d_model:int, num_heads:int, batch_size:int):
        super(MHAttention,self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.depth = self.d_model // self.num_heads
        assert self.d_model % self.num_heads == 0

        
        self.sqrt_depth = tf.cast(self.depth, dtype=tf.float32)

        self.query = tf.keras.layers.Dense(self.d_model)
        self.key = tf.keras.layers.Dense(self.d_model)
        self.value = tf.keras.layers.Dense(self.d_model)
        self.outweight = tf.keras.layers.Dense(self.d_model)
        self.softmax = tf.keras.layers.Softmax(axis=-1)


    def split_heads(self, input):
        # if self.batch_size == None:
        #     batch_size = tf.shape(input)[0]
        # else: 
        #     batch_size = self.batch_size
        input = tf.reshape(input,(self.batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(input, perm=[0,2,1,3])


    def __call__(self, input):
        # if self.batch_size == None:
        #     batch_size = tf.shape(input)[0]
        # else: 
        #     batch_size = self.batch_size
        query = self.query(input)
        key = self.key(input)
        value = self.value(input)

        query_splitted = self.split_heads(query)
        key_splitted = self.split_heads(key)
        value_splitted = self.split_heads(value)

        q_mat_k = tf.matmul(query_splitted, key_splitted, transpose_b=True)
        q_mat_k = q_mat_k / self.sqrt_depth

        q_mat_k_soft = self.softmax(q_mat_k)
        attention_score = tf.matmul(q_mat_k_soft, value_splitted)
        attention_score = tf.transpose(attention_score, perm=[0,2,1,3])
        attention_score = tf.reshape(attention_score, (self.batch_size, -1, self.d_model))

        return self.outweight(attention_score)


class MyEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model:int, num_heads:int, batch_size:int):
        super(MyEncoder,self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.multi_head_attention = MHAttention(self.d_model, self.num_heads, self.batch_size)

        self.dense1 = tf.keras.layers.Dense(d_model)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.act1 = tf.keras.layers.Activation('relu')

    def __call__(self, input):
        a = self.multi_head_attention(input)
        con = tf.concat([input,a], axis=-1)
        o1 = self.dense1(con)
        o1 = self.layer_norm(o1)
        o1 = self.act1(o1)       

        return o1

class Res1(tf.keras.layers.Layer):
    def __init__(self,filters:int,kernel_size:int,padding:str,activation:str, flag_res:bool=True):
        super(Res1,self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.FLAG_RES = flag_res

        self.conv1 = Conv1D(self.filters, kernel_size=self.kernel_size, padding=self.padding)
        self.batch1 = BatchNormalization()
        self.act1 = Activation(self.activation)

        self.conv2 = Conv1D(self.filters, kernel_size=self.kernel_size, padding=self.padding)
        self.batch2 = BatchNormalization()
        self.act2 = Activation(self.activation)

        self.conv3 = Conv1D(self.filters, kernel_size=self.kernel_size, padding=self.padding)
        self.batch3 = BatchNormalization()
        self.act3 = Activation(self.activation)

        self.pool = MaxPool1D(strides=2)


    def __call__(self, input):
        x1 = self.conv1(input)
        x2 = self.batch1(x1)
        x3 = self.act1(x2)

        x4 = self.conv2(x3)
        x5 = self.batch2(x4)
        x6 = self.act2(x5)

        x7 = self.conv3(x6)
        x8 = self.batch3(x7)
        x9 = self.act3(x8)

        if self.FLAG_RES:
            x9 = tf.add(x9, x3)
        x10 = self.pool(x9)

        return x10


class MobileDense1(tf.keras.layers.Layer):
    ## the number of filters must be twice of input channels because the output will be concatenated. 
    def __init__(self,filters:int, kernel_size:int=3, padding:str='same', activation:str='relu', depth_mul:int=4):
        super(MobileDense1,self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.depth_mul = depth_mul

        if self.filters & 1:
            raise Exception("from MobileDense1: the parameter named 'filters' must be even number")

        self.conv1 = DepthwiseConv2D(self.kernel_size, depth_multiplier=self.depth_mul, padding=self.padding)
        self.bn1 = BatchNormalization()
        self.act1 = Activation(self.activation)
        self.conv2 = Conv2D(self.filters//2, 1, padding=self.padding)
        self.bn2 = BatchNormalization()
        self.act2 = Activation(self.activation)
        self.pool = MaxPool2D(strides=2)
        
    def __call__(self, input):
        if input.shape[-1] != self.filters//2:
            raise Exception("from MobileDense1: the input channels must be half of the 'filters' parameter")

        x1 = self.conv1(input)
        x2 = self.bn1(x1)
        x3 = self.act1(x2)
        x4 = self.conv2(x3)
        x5 = self.bn2(x4)
        x6 = self.act2(x5)
        x7 = tf.concat([input, x6],axis=-1)
        
        return self.pool(x7)


if "__main__" == __name__:


    # input_layer = Input(shape=(64000,1), batch_size=BATCH_SIZE)

    # res1 = Res1(filters, 3, 'same', ACTIVATION_FN)(input_layer)
    # ## 32000, 32

    # res2 = Res1(filters*2, 3, 'same', ACTIVATION_FN)(res1)
    # ## 16000, 64

    # res3 = Res1(filters*4, 3, 'same', ACTIVATION_FN)(res2)
    # ## 8000, 128

    # res4 = Res1(filters*8, 3, 'same', ACTIVATION_FN)(res3)
    # ## 4000, 256

    # res5 = Res1(filters*16, 3, 'same', ACTIVATION_FN)(res4)
    # ## 2000, 512

    # conv1 = Conv1D(filters *16, kernel_size=3, padding='same')(res5)
    # ## 2000, 512
    # tr1 = tf.transpose(conv1,perm=[0,2,1])
    # # print("input_layer, ", res10)
    # enc1 = MyEncoder(2000,8,BATCH_SIZE)(tr1)

    # flat1 = Flatten()(enc1)
    # batch6 = BatchNormalization()(flat1)
    # dense1 = Dense(200, activation='relu')(batch6)
    # batch7 = BatchNormalization()(dense1)
    # dense2 = Dense(20, activation='relu')(batch7)
    # # dense2 = Dense(20, activation='relu')(dense1)
    # output_layer = Dense(1, activation='sigmoid')(dense2)

    # model = Model(input_layer, output_layer)

    # model.summary()

    
    input_layer = Input(shape=(64000,1), batch_size=BATCH_SIZE)
    conv = Conv1D(320,3, padding='same')(input_layer)
    bn = BatchNormalization()(conv)
    act = Activation('relu')(bn)
    reshaped1 = tf.reshape(act, (-1,800,800,32))
    md1 = MobileDense1(64)(reshaped1)
    ## 400, 64

    md2 = MobileDense1(128)(md1)
    ## 200, 128

    md3 = MobileDense1(256)(md2)
    ## 100, 256

    md4 = MobileDense1(512)(md3)
    ## 50, 512

    reshaped2 = tf.reshape(md4,(-1,2500,512))
    enc1 = MyEncoder(2500,10,BATCH_SIZE)(reshaped2)

    flat1 = Flatten()(enc1)
    batch6 = BatchNormalization()(flat1)
    dense1 = Dense(200, activation='relu')(batch6)
    batch7 = BatchNormalization()(dense1)
    dense2 = Dense(20, activation='relu')(batch7)
    # dense2 = Dense(20, activation='relu')(dense1)
    output_layer = Dense(1, activation='sigmoid')(dense2)

    model = Model(input_layer, output_layer)

    model.summary()