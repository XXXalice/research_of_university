from urllib.request import urlopen, Request
from urllib.parse import quote
from mimetypes import guess_extension
import os
import sys
from bs4 import BeautifulSoup
from time import sleep
from param import base_param
from wikipedia.near_word import near_word

class Fetcher:
    """
    受け取ったurlを指定されたuaでリクエストし、html、MIMEタイプとエンコードをもらう
    """
    def __init__(self, ua=''):
        self.ua = ua

    def fetch(self, url):
        req = Request(url, headers={'User-Agent': self.ua})
        try:
            with urlopen(req, timeout=2) as p:
                content = p.read()
                mime = p.getheader('Content-Type')
        except:
            print('Error in fetching.')
            return None, None

        return content, mime

def url_search(keyword, n=base_param.IMAGE_NUM):
    code = '&ei=UTF-8'
    img_link_elem = [] #対象データのクエリを格納する
    alt_img_link_elem = [] #対象データ「ではない」クエリを格納する
    near_words = near_word(keyword)

    #検索するキーワード群
    #[0]は対象データ、それ以降は対象以外のデータ
    search_words = []
    search_words.append(keyword)
    search_words += near_words

    for i, search_word in enumerate(search_words):
        #こっからはyahooで一度に引っ張ってこれる数が60枚までなので苦肉の策でスクレイピング

        #対象データを収集する場合
        if i == 0:
            if n >= 60:
                #検索回数
                iter_num = int(n / 60)
                #残りの枚数
                rem = n % 60
                last_num = 0
                for i in range(2,iter_num+2):
                    last_num = i
                    url = ('https://search.yahoo.co.jp/image/search?n=60&p={}{}' + code).format(quote(search_word), i)
                    byte_content, mime = fetcher.fetch(url)
                    soup = BeautifulSoup(byte_content.decode('UTF-8'), 'html.parser')
                    elem = soup.find_all('a', attrs={'target': 'imagewin'})
                    img_link_elem.extend(elem[60:])
                if rem != 0:
                    url = ('https://search.yahoo.co.jp/image/search?n={}&p={}{}' + code).format(rem, quote(search_word), last_num)
                    byte_content, mime = fetcher.fetch(url)
                    soup = BeautifulSoup(byte_content.decode('UTF-8'), 'html.parser')
                    add_elem = soup.find_all('a', attrs={'target': 'imagewin'})
                    img_link_elem.extend(add_elem[rem:])
            else: #要求画像が60枚未満の時
                url = ('https://search.yahoo.co.jp/image/search?n={}&p={}' + code).format(n, quote(search_word))
                byte_content, mime = fetcher.fetch(url)
                soup = BeautifulSoup(byte_content.decode('UTF-8'), 'html.parser')
                elem = soup.find_all('a', attrs={'target': 'imagewin'})
                img_link_elem.extend(elem[n:])

        #対象「ではない」データを収集する場合
        else:
            url = ('https://search.yahoo.co.jp/image/search?n={}&p={}'.format(base_param.ALT_IMAGE_NUM, quote(search_word)))
            byte_content, mime = fetcher.fetch(url)
            soup = BeautifulSoup(byte_content.decode('UTF-8'), 'html.parser')
            elem = soup.find_all('a', attrs={'target': 'imagewin'})
            alt_img_link_elem.extend(elem[base_param.ALT_IMAGE_NUM:])

    img_urls = [e.get('href') for e in img_link_elem if e.get('href').startswith('http')]
    alt_img_urls = [e.get('href') for e in alt_img_link_elem if e.get('href').startswith('http')]

    print(len(img_urls), len(alt_img_urls))

    return (img_urls, alt_img_urls) #タプルで返すよ〜

def image_collector_in_url(urls, fname):
    d = './img/'+str(fname)
    if not os.path.exists(d):
        os.mkdir(d)
        os.mkdir(d+'/train')
        os.mkdir(d+'/test')

    train_num = int(len(urls)*(base_param.TRAIN_PAR/100)) #訓練画像の枚数
    count = 0

    for i, img_url in enumerate(urls):
        sleep(0.1) #礼儀
        img, mime = fetcher.fetch(img_url)
        if not mime or not img:
            continue
        try:
            ext = guess_extension(mime.split(';')[0]) #拡張子
            if ext in ('.jpeg','.jpg','.png','.jpe'):
                ext = '.png' #全部pngで統一
            else:
                continue
        except Exception as e:
            sys.stdout.write(e,mime)
            continue
        flag = 0 if i+1 < train_num else 1
        flag2 = '/train/' if flag == 0 else '/test/'
        file = os.path.join(d+flag2, str(flag)+'_{number:03}{ext}'.format(number=i+1, ext=ext))
        with open(file, mode='wb') as f:
            f.write(img)
            print('download success:',img_url)
            count += 1

    return count

if __name__ == '__main__':

    fetcher = Fetcher(base_param.UA)
    print('欲しいモデルのワードを入力してください')
    keyword = input('>> ')
    urls = url_search(keyword) #タプルがかえってくるよ〜
    print('生成するフォルダ名を入力してください（なるべく英数で）')
    fname = input('>> ')
    download_count = image_collector_in_url(urls[0]+urls[1], fname)
    print('ダウンロード枚数:', download_count)