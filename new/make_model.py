from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Flatten, Activation, Convolution2D, MaxPooling2D
from img2array import Img2Array
import os

def make_model(shape):
    model = Sequential()
    model.add(Convolution2D(
        filters=14,
        kernel_size=3,
        strides=2,
        border_mode='same',
        input_shape=shape
    ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(
        filters=28,
        kernel_size=3,
        strides=3,
        border_mode='same',
    ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy']
    )

    return model

def training(model, image, label):
    return model.fit(image, label, nb_epoch=100, batch_size=10)


if __name__ == '__main__':
    img2array = Img2Array()
    try:
        train_image, train_label = img2array.make_array(train=True)
        test_image, test_label = img2array.make_array(train=False)
        print('info: 画像の数値化に成功しました train{}枚 test{}枚'.format(len(train_image), len(test_image)))
        print('info: train shape {}'.format(train_image.shape[1:]))
    except Exception as e:
        print(e)
        exit()

    try:
        model = make_model(train_image.shape[1:])
        print('info: モデルの作成に成功しました')
        trained_model = training(model, train_image, train_label)
        print('info: モデルの学習に成功しました')
    except Exception as e:
        print(e)
        exit()

    score = model.evaluate(test_image, test_label, verbose=100)
    print('loss=', score[0])
    print('accuracy=', score[1])

    #モデルをセーブ
    if not os.path.exists('./src'):
        os.mkdir('./src')
    model.save('./src/test_model.h5')