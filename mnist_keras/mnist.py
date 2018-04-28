from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
import os

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#データをfloat32型に変換して正規化する
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float')
X_train /= 255
X_test /= 255
#ラベルデータを0~9までのカテゴリに表す配列を変換
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

#モデルの構造を定義
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0,2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

#モデルを構築
model.compile(
    loss = 'categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)

#データで訓練
hist = model.fit(X_train, y_train)

#テストデータを用いて訓練する
score = model.evaluate(X_test, y_test, verbose=100)
print('loss=', score[0])
print('accuracy=', score[1])

#モデルをセーブ
json_string = model.to_json()
if not os.path.exists('./src'):
    os.mkdir('./src')
model.save('./src/mnist_model.h5')