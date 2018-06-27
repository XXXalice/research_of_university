from keras.applications import inception_v3
from keras import backend as K

K.set_learning_phase(0)

model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

#勾配率
layer_contributions = {
    'mixed2':0.4,
    'mixed3':3.,
    'mixed4':2.,
    'mixed5':1.5,
}

layer_dict = dict([(layer.name, layer) for layer in model.layers])

loss = K.variable(0.)
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]

    activation = layer_dict[layer_name].output

    scaling = K.prod(K.cast(K.shape(activation), 'float32'))

    loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling

#生成された画像（dream）を保持するテンソル
dream = model.input

#ドリームの損失関数の勾配を計算
grads = K.gradients(loss, dream)[0]

#勾配を正規化する
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

#入力画像に基づいて損失と勾配の値を取得するkeras関数を設定
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs=outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x

import numpy as np
import scipy
import os
import base_param_test as bpt
from keras.preprocessing import image

#勾配上昇方のステップサイズ
step = 0.01
#勾配上昇方を実行する尺度の数
num_octave = 3
#尺度間の拡大率
octave_scale = 1.4
#尺度ごとの上昇ステップの数
iterations = 20

#損失値が10を超えた場合は見た目が酷くなるのを避けるために勾配上昇法を中止
max_loss = 20.

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

#画像を開いてサイズを変更し、inception V3が処理できるテンソルに変換
def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
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

img = preprocess_image(base_image)

original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)

successive_shapes = successive_shapes[::-1]

original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss
                          )

    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img
    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    #save_img(img, fname='dream_at_scale_' + str(shape) + '.png')
save_img(img, fname=base_image_path + '/' + bpt.USE_IMAGE + '_nightmare.jpg')