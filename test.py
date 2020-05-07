import numpy as np
import os
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from model import *
from set_data import *
import random
import cv2
(X_train, Y_train), (X_test, Y_test) = load_data()#加载mnist
X_train = np.vstack((X_train,X_test))
Y_train = np.hstack((Y_train,Y_test))

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

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

def char_decode(label_encode): 
    return [''.join([idx_map[column] for column in row]) for row in label_encode]

def fit_keras_channels(batch):
    if K.image_data_format() == "channels_first":
        batch = batch.reshape(batch.shape[0],1,img_h,img_w)
    else:
        batch = batch.reshape(batch.shape[0],img_h,img_w,1)        
    return batch
def generate_test_data(batch_size):
    while True:
        test_labels_batch = []
        test_imgs_batch = []
        length = random.randint(min_length, max_length)
        for _ in range(batch_size):
            img,char= test_create_img(length)
            img_gray = cv2.resize(img, (160, 60))
            img = img_gray/255.
            img = np.asarray(img)
            test_labels_batch.append(char)
            test_imgs_batch.append(img)
        yield([np.array(test_imgs_batch), np.array(test_labels_batch)])


def test(test_batch_size=32, test_iter_num=100):
    error_cnt = 0
    sample_num = 0
    loss_pred_all = None
    iterator = generate_test_data(test_batch_size)
    for _ in range(test_iter_num):
        test_imgs_batch, test_labels_batch = next(iterator)

        test_labels_encode_batch = []
        for label in test_labels_batch:
            label = [char_map[i] for i in label]
            test_labels_encode_batch.append(label)  

        test_imgs_batch= fit_keras_channels(test_imgs_batch)
        labels_pred = model.predict_on_batch(np.array(test_imgs_batch))
        labels_pred = char_decode(labels_pred)  
        
        for label, label_pred in zip(test_labels_batch, labels_pred):
            if label != label_pred:
                error_cnt += 1
                #print(f'{label} -> {label_pred}')
        
    
        ### 保存测试数据以便绘制loss的概率密度函数
        loss_pred = fit_model.predict([test_imgs_batch, np.array(test_labels_encode_batch)])
        if loss_pred_all is None:
            loss_pred_all = loss_pred
        else:
            loss_pred_all = np.vstack([loss_pred_all, loss_pred])
        sample_num += len(loss_pred)

    print(f'总样本数：{test_batch_size * test_iter_num} | '
          f'错误数：{error_cnt} | '
          f'准确率：{1 - error_cnt / test_batch_size / test_iter_num}')

    ### 绘制loss的概率密度函数
    sns.distplot(loss_pred_all)
    plt.title(f'mean: {loss_pred_all.mean():.3f} | '
              f'max: {loss_pred_all.max():.3f} | '
              f'median: {np.median(loss_pred_all):.3f}\n'
              f'总样本数：{sample_num} | '
              f'错误数：{error_cnt} | '
              f'准确率：{1 - error_cnt / sample_num:.3f}', fontsize = 20)
    plt.xlabel('loss', fontsize = 20)
    plt.ylabel('PDF', fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
def real_test():
    test_imgs_batch = []
    filelist = []
    for file in os.listdir("./test"):
        img = cv2.imread("./test/"+file)
        h_resize = 60
        w_resize = 160
        img_gray = cv2.cvtColor(cv2.resize(img, (w_resize, h_resize)), cv2.COLOR_BGR2GRAY) # 缩小图片固定宽度为32，并转为灰度图
        img_gray = (255-img_gray)/255.
        test_imgs_batch.append(img_gray)
        
        filelist.append(file[:-4]) 
    test_imgs_batch= fit_keras_channels(np.array(test_imgs_batch))
    labels_pred = model.predict_on_batch(test_imgs_batch)
    labels_pred = char_decode(labels_pred)
    print("真实为：",filelist,"\n识别为：",labels_pred)
    imgs = os.listdir("./test")
    for i in range(0,len(os.listdir("./test"))-1):
        # 调用cv.putText()添加文字
        img = cv2.imread("./test/"+imgs[i])
        text = labels_pred[i]
        AddText = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
        # 使用白色填充图片区域,默认为黑色
        AddText.fill(255)
        cv2.putText(AddText, text, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 0), 5)
        # 将原图片和添加文字后的图片拼接起来
        res = np.vstack([img, AddText])
        # 显示拼接后的图片
        cv2.imwrite('./save/'+imgs[i], res) 
if __name__ == '__main__':

    fit_model = model_test01(img_w,img_h,True)
    model = model_test01(img_w,img_h,False)
    fit_model.load_weights(f'./model/fitmodel.h5')
    model.load_weights(f'./model/fitmodel.h5')
    test(test_batch_size = 32, test_iter_num=100)
    real_test()