
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten,GRU
from tensorflow.keras.layers import Input,Activation
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Reshape,Bidirectional,Dropout
from tensorflow.keras.layers import TimeDistributed,BatchNormalization,Concatenate,add
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adadelta
from set_data import *
class StopTraining(keras.callbacks.Callback):
    def __init__(self, thres):
        super(StopTraining, self).__init__()
        self.thres = thres
    def on_epoch_end(self, batch, logs={}):
        if logs.get('loss') < self.thres:
            self.model.stop_training = True
def ctc_loss(args):
    return K.ctc_batch_cost(*args)
def ctc_decode(softmax):
    return K.ctc_decode(softmax, K.tile([K.shape(softmax)[1]], [K.shape(softmax)[0]]))[0][0]
def get_model(img_w,img_h,train):
    input_shape = (img_h, img_w, 1)
    inputs = Input(shape =input_shape,name = "inputs")
    conv1 = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same',name='conv1')(inputs)
    conv1 = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same',name='conv1_')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool1')(conv1)
    conv2 = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same',name='conv2')(pool1)
    conv2 = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same',name='conv2_')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool2')(conv2)
    conv3 = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same',name='conv3')(pool2)
    conv4 = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same',name='conv4')(conv3)
    conv4 = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same',name='conv4_')(conv4)
    pool3 = MaxPooling2D(pool_size=(2,2),strides=(2,1),padding='valid',name='pool3')(conv4)

    conv5 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same',name='conv5')(pool3)
    conv6 = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same',name='conv6')(conv5)
    pool4 = MaxPooling2D(pool_size=(2,2),strides=(2,1),padding='valid',name='pool4')(conv6)
    conv7 = Conv2D(512,kernel_size=(2,2),activation='relu',padding='valid',name='conv7')(pool4)
    conv7 = MaxPooling2D(pool_size=(2,1),strides=(2,1),padding='valid',name='pool7')(conv7)
    m = Permute((2,1,3),name='permute')(conv7)
    timedistrib = TimeDistributed(Flatten(),name='timedistrib')(m)
    rnnunit = 256
    bgru1 = Bidirectional(GRU(rnnunit,return_sequences=True,implementation=2),name='bgru1')(timedistrib)
    dense = Dense(rnnunit,name='bgru1_out',activation='linear',)(bgru1)
    bgru2 = Bidirectional(GRU(rnnunit,return_sequences=True,implementation=2),name='bgru2')(dense)
    outputs = Dense(len(chars) + 1,name='bgru2_out',activation='softmax')(bgru2)

    labels_input = Input([None], dtype='int32')

    input_length = Lambda(lambda x: K.tile([[K.shape(x)[1]]], [K.shape(x)[0], 1]))(outputs)
    label_length = Lambda(lambda x: K.tile([[K.shape(x)[1]]], [K.shape(x)[0], 1]))(labels_input)
    output = Lambda(ctc_loss)([labels_input, outputs, input_length, label_length])
    fit_model = Model(inputs=[inputs, labels_input], outputs=output)
    ctc_decode_output = Lambda(ctc_decode)(outputs)
    model = Model(inputs=inputs, outputs=ctc_decode_output)
    if train:
        return fit_model
    else:
        return model
