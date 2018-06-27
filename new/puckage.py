import img_collector as imgc
import make_model as mkmodel

class Puckage:
    def __init__(self, name, keyword):
        self.name = name
        self.keyword = keyword

    def img_collect(self):
        urls = imgc.url_search(self.keyword)
        download_count = imgc.image_collector_in_url(urls, self.name)
        print('info: 合計{}枚の画像の収集に成功しました'.format(str(download_count)))

    def makemodel(self):

        try:
            mkmodel.make_to_test(self.name)
        except Exception as e:
            print(e)
            exit()