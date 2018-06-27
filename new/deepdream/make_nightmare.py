from keras import layers
from keras import models
from keras import backend as K
import sys
sys.path.append('../')
from param import base_param as bp

import numpy as np
from . import img_processing as imgprc
from . import base_param_test as bpt

K.set_learning_phase(0)

model = models.Sequential()

model.add(layers.Conv2D(filters=14,
                        kernel_size=3,
                        strides=2,
                        border_mode='same',
                        activation='relu',
                        input_shape=((14, 14, 3))
                        ))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(filters=28,
                        kernel_size=3,
                        strides=2,
                        border_mode='same',
                        activation='relu',
                        ))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

model.load_weights(filepath=bp.MODEL_PATH + bp.MODEL_DEEPDREAM, by_name=True)

layer_contributions = {
    'conv2d_1': 0.2,
    'conv2d_2': 3.,
    'dense_1': 2.,
}

layer_dict = dict([(layer.name, layer) for layer in model.layers])
loss = K.variable(0.)
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))

    loss += coeff / scaling

dream = model.input

grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

#勾配上昇方のステップサイズ
step = 0.01
#勾配上昇方を実行する尺度の数
num_octave = 2
#尺度間の拡大率
octave_scale = 1.4
#尺度ごとの上昇ステップの数
iterations = 20

#損失値が10を超えた場合は見た目が酷くなるのを避けるために勾配上昇法を中止
max_loss = 50.



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
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x

img = imgprc.preprocess_image(imgprc.base_image)

original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)

successive_shapes = successive_shapes[::-1]

original_img = np.copy(img)
shrunk_original_img = imgprc.resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = imgprc.resize_img(img, (bp.IMAGE_SIZE, bp.IMAGE_SIZE))
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss
                          )

    upscaled_shrunk_original_img = imgprc.resize_img(shrunk_original_img, (bp.IMAGE_SIZE, bp.IMAGE_SIZE))
    same_size_original = imgprc.resize_img(original_img, (14, 14))
    lost_detail = same_size_original - upscaled_shrunk_original_img
    img += lost_detail
    shrunk_original_img = imgprc.resize_img(original_img, (14, 14))
    #save_img(img, fname='dream_at_scale_' + str(shape) + '.png')
imgprc.save_img(img, fname=imgprc.base_image_path + '/' + bpt.USE_IMAGE + '_nightmare.jpg')