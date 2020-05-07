
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
    inner = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    inner = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    inner = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(inner)
    inner = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    inner = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(inner)
    inner = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(inner)
    inner = MaxPooling2D(pool_size=(2, 2))(inner)
    inner = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(inner)
    inner = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(inner)
    inner = MaxPooling2D(pool_size=(2, 1))(inner)
    inner = Permute((2, 1, 3))(inner)
    inner = TimeDistributed(Flatten())(inner)
    gru = Bidirectional(GRU(128,return_sequences=True))(inner)
    gru = Bidirectional(GRU(128,return_sequences=True))(gru)
    outputs =TimeDistributed(Dense(len(chars) + 1, activation='softmax'))(gru)

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
