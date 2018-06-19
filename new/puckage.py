import img_collector as imgc
from img2array import Img2Array
import make_model as mkmodel

class Puckage:
    def __init__(self):
        pass

    def img_collect(self):
        keyword = str(input('モデルを生成するキーワードを入力して下さい>> '))
        urls = imgc.url_search(keyword)
        name = str(input('フォルダ名、モデル名に使用する英数名を入力して下さい>> '))
        download_count = imgc.image_collector_in_url(urls, name)
        print('info: 合計{}枚の画像の収集に成功しました'.format(str(download_count)))
        return name

    def makemodel(self,name):
        import os
        img2array = Img2Array(name)
        try:
            train_image, train_label = img2array.make_array(train=True)
            test_image, test_label = img2array.make_array(train=False)
            print('info: 画像の数値化に成功しました train{}枚 test{}枚'.format(len(train_image), len(test_image)))
            print('info: train shape {}'.format(train_image.shape[1:]))
        except Exception as e:
            print(e)
            exit()

        try:
            model = mkmodel.make_model(train_image.shape[1:])
            print('info: モデルの作成に成功しました')
            trained_model = mkmodel.training(model, train_image, train_label)
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
        model.save('./src/{}_model.h5'.format(name))

        return 'info: モデルの保存に成功しました'

