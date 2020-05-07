import numpy as np
import cv2
import random
import os
def load_data(path="./mnist.npz"):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
(X_train, Y_train), (X_test, Y_test) = load_data()#加载mnist
X_train = np.vstack((X_train,X_test))
Y_train = np.hstack((Y_train,Y_test))

def add_img(leftgray,rightgray):
    num = random.randint(0,5)
    final_matrix = np.zeros((28,leftgray.shape[1]+28-2*num), np.uint8)
    final_matrix[0:28, 0:leftgray.shape[1]-num] = leftgray[0:28, 0:leftgray.shape[1]-num]
    final_matrix[0:28, leftgray.shape[1]-num:leftgray.shape[1]+28-2*num] = rightgray[0:28, num:leftgray.shape[1]]
    return final_matrix
def test_create_img(num):
    for i in range(1, num):  
        a = random.randint(0,69999)# 迭代 0 到 69999 之间的数字
        if(i == 1):
            b = random.randint(0,69999)
            img =add_img(X_train[a],X_train[b])
            label = str(Y_train[a]) + str(Y_train[b])
        else:
            img = add_img(img,X_train[a])
            label = label + str(Y_train[a])
    return img,label
img ,label = test_create_img(10)
print(label)
cv2.imshow('im',img)
cv2.waitKey()
cv2.destroyAllWindows()