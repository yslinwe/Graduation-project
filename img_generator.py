import cv2
import os, random
import numpy as np
import string
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K

chars = string.digits # 验证码字符集
char_map = {chars[c]: c for c in range(len(chars))} # 验证码编码（0到len(chars) - 1)

class ImageGenerator:
    def __init__(self, X_train,Y_train,img_w, img_h,min_length,max_length,number):
        self.img_h = img_h
        self.img_w = img_w
        self.min_length = min_length
        self.max_length = max_length
        self.X_train = X_train
        self.Y_train = Y_train
        self.number = number
        self.aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
        width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
        horizontal_flip=True, fill_mode="nearest")
    def add_img(self,leftgray,rightgray):
        num = random.randint(0,5)
        final_matrix = np.zeros((28,leftgray.shape[1]+28-2*num), np.uint8)
        final_matrix[0:28, 0:leftgray.shape[1]-num] = leftgray[0:28, 0:leftgray.shape[1]-num]
        final_matrix[0:28, leftgray.shape[1]-num:leftgray.shape[1]+28-2*num] = rightgray[0:28, num:leftgray.shape[1]]
        return final_matrix
    def test_create_img(self,num):
        for i in range(1, num):  
            a = random.randint(0,69999)# 迭代 0 到 69999 之间的数字
            if(i == 1):
                b = random.randint(0,69999)
                img =self.add_img(self.X_train[a],self.X_train[b])
                label = str(self.Y_train[a]) + str(self.Y_train[b])
            else:
                img = self.add_img(img,self.X_train[a])
                label = label + str(self.Y_train[a])
        return img,label

    def load_img(self):
        labels = {length: [] for length in range(self.min_length, self.max_length + 1)} 
        imgs = {length: [] for length in range(self.min_length, self.max_length + 1)}
        labels_encode = {length: [] for length in range(self.min_length, self.max_length + 1)}
        for length in range(self.min_length, self.max_length + 1):
        #length = 4
            for _ in range(self.number):#生成图片个数
                img,label = self.test_create_img(length)
                labels[length].append(label)
                
                img_gray = cv2.resize(img, (self.img_w, self.img_h))
                img = img_gray/255.
                img = np.asarray(img)
                imgs[length].append(img)

                label = [char_map[i] for i in label]
                labels_encode[length].append(label)

        return imgs, labels, labels_encode
    def fit_keras_channels(self,batch):
        if K.image_data_format() == "channels_first":
            batch = batch.reshape(batch.shape[0],1,self.img_h,self.img_w)
        else:
            batch = batch.reshape(batch.shape[0],self.img_h,self.img_w,1)        
        return batch

    def generate_data(self,imgs, labels_encode, batch_size):
        imgs = {length: np.array(imgs[length]) for length in range(self.min_length, self.max_length + 1)} # 图片BGR数据字典{长度：BGR数据数组}
        labels_encode = {length: np.array(labels_encode[length]) for length in range(self.min_length, self.max_length + 1)} # 验证码真实标签{长度：标签数组}
        while True:
            length = random.randint(self.min_length, self.max_length)
            #length = 4
            test_idx = np.random.choice(range(len(imgs[length])), batch_size)
            batch_imgs = np.array(imgs[length][test_idx],dtype=np.float32)
            batch_imgs= self.fit_keras_channels(batch_imgs)
            batch_labels = np.array(labels_encode[length][test_idx])
            batch_imgs ,batch_labels= self.aug.flow(batch_imgs,batch_labels,batch_size = batch_size).next()
            yield ([batch_imgs, batch_labels], None)# 元组的第一个元素为输入，第二个元素为训练标签，即自定义loss函数时的y_true
