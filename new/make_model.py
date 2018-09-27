from img2array import Img2Array
from param import base_param as bp
from models import Models

def i2a(name):
    img2array = Img2Array(name)
    try:
        train_image, train_label = img2array.make_array(train=True)
        test_image, test_label = img2array.make_array(train=False)
        print('info: 画像の数値化に成功しました train{}枚 test{}枚'.format(len(train_image), len(test_image)))
        print('info: train shape {}'.format(train_image.shape[1:]))
    except Exception as e:
        print(e)
        exit()
    return train_image, train_label, test_image, test_label

def make_model(train_image, train_label, test_image, test_label, name):
    try:
        m = Models(shape=train_image.shape[1:], name=name)
        model = exec('m.' + bp.USE_MODEL + '(train_image, train_label, test_image, test_label)')
        print('info: input_shape:{}'.format(train_image.shape[1:]))
    except Exception as e:
        print(e)
        exit()


def make_to_test(name):
    train_image, train_label, test_image, test_label = i2a(name)
    trained_model = make_model(train_image, train_label, test_image, test_label, name)
    return trained_model