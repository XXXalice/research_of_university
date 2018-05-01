import cv2
import numpy as np
from param import base_param

class Img2Array:
    """
    画像をnumpy配列に変換
    """
    def __init__(self,img_path='./img/'):
        self.img_path = img_path

    def make_array(self,train):
        """
        :param train: 訓練データだったらT、テストデータだったらF
        :return: 訓練またはテストのnumpy配列
        """
        if train == True:
            txt_file = 'train_list.txt'
            img_folder = 'train/'
        else:
            txt_file = 'test_list.txt'
            img_folder = 'test/'
        with open(self.img_path + txt_file, 'r') as f:
            images = []
            labels = []
            for line in f:
                #末尾の空白文字、つまり改行をstrip
                line = line.rstrip()
                #[画像ファイル名,ラベル]のリスト化
                l = line.split()
                img = cv2.imread(self.img_path + img_folder + l[0])
                img = cv2.resize(img, (28, 28))
                #画像を一次元化して正規化
                images.append(img.flatten().astype(np.float32)/255.0)
                tmp = np.zeros(base_param.NUM_CLASSES)
                tmp[int(l[1])] = 1
                labels.append(tmp)
            images = np.asarray(images)
            labels = np.asarray(labels)

        return images, labels