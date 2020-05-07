import string
import numpy as np
img_w = 160
img_h = 60
min_length = 4
max_length = 7
train_num = 10
chars = '0123456789' # 验证码字符集
char_map = {chars[c]: c for c in range(len(chars))} # 验证码编码（0到len(chars) - 1)
idx_map = {value: key for key, value in char_map.items()} # 编码映射到字符
idx_map[-1] = '' # -1映射到空

def load_data(path="./mnist.npz"):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)