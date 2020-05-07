from img_generator import ImageGenerator 
from model import get_model,StopTraining
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.models import Model
import numpy as np
from set_data import *


(X_train, Y_train), (X_test, Y_test) = load_data()#加载mnist
X_train = np.vstack((X_train,X_test))
Y_train = np.hstack((Y_train,Y_test))

train_img = ImageGenerator(X_train,Y_train,img_w,img_h,min_length,max_length,train_num)
X_trains,Y_label,Y_trains = train_img.load_img()

fit_model = get_model(img_w,img_h,True)

adadelta = Adadelta(lr=0.05)

fit_model.compile(
    loss=lambda y_true, y_pred: y_pred,
    optimizer=adadelta)
fit_model.summary()

fit_model.fit_generator(
generator = train_img.generate_data(X_trains, Y_trains, 32), 
epochs=1000, 
steps_per_epoch=100, 
callbacks=[StopTraining(0.01)],
verbose=1
)
fit_model.save(f'./model/fit_model_test.h5')
