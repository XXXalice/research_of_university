from keras import models
from keras import optimizers
from keras import layers
import os
import base_param as bp

class Models:
    def __init__(self,shape,name):
        self.shape = shape
        self.name = name

    def original1(self, train_image, train_label, test_image, test_label):
        model = models.Sequential()

        model.add(layers.Conv2D(filters=32,
                                kernel_size=(3, 3),
                                border_mode='same',
                                activation='relu',
                                input_shape=(self.shape)
                                ))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(filters=64,
                                kernel_size=(3, 3),
                                border_mode='same',
                                activation='relu',
                                ))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(2, activation='softmax'))

        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizers.RMSprop(),
            metrics=['accuracy']
        )

        print('info: モデルの作成に成功しました')

        model.fit(train_image, train_label, batch_size=10, epochs=15)

        print('info: モデルの学習に成功しました')

        score = model.evaluate(test_image, test_label, verbose=100)
        print('loss=', score[0])
        print('accuracy=', score[1])

        if not os.path.exists(bp.MODEL_PATH):
            os.mkdir(bp.MODEL_PATH)
        model.save_weights(filepath= bp.MODEL_PATH + '/{}_model.h5'.format(self.name))

        print('info: モデルの保存に成功しました')