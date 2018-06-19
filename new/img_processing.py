import scipy
from keras.preprocessing import image
import numpy as np
import os
from . import base_param_test as bpt
from keras import backend as K


base_image_path = os.path.join('./img',bpt.USE_FOLDER)
base_image = os.path.join('./img', bpt.USE_FOLDER, bpt.USE_IMAGE+'.jpg')


#画像のサイズを変更
def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0])/img.shape[1],
               float(size[1])/img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)

#画像を保存
def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)

#画像を開いてサイズを変更し、処理できるテンソルに変換
def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

#テンソルを有効な画像に変換
def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1,2,0))
    else:
        #inception_v3.preprocess_inputによって実行された前処理を元に戻す
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x
