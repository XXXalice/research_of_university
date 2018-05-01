from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation
from img2array import Img2Array

if __name__ == '__main__':
    img2array = Img2Array()
    try:
        train_image, train_label = img2array.make_array(train=True)
        test_image, test_label = img2array.make_array(train=False)
    except Exception as e:
        print(e)
        exit()
    print('ok')