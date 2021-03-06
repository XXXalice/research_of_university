USE_FOLDER = 'anime'

#モデル構築の際使用するCNN
USE_MODEL = 'original1'

#訓練データの割合（デフォで30% 訓練:テスト 7:3）
TRAIN_PAR = 70

#分類数
NUM_CLASSES = 2

#画像の一辺のサイズ
IMAGE_SIZE = 100

#画像収集に使用する際のユーザーエージェント偽装（デフォはpython）
UA = ''

#対象の画像の収集枚数
IMAGE_NUM = 50

#対象「ではない」画像の1グループの枚数（1グループ*5枚の画像を収集します）
ALT_IMAGE_NUM = 10

#モデルのパス
MODEL_PATH = '../src/'

#deepdreamで使用するモデル
MODEL_DEEPDREAM = ''

#水増し用フォルダ
GENARATE_FOLDER = ['./img/abe/train','./img/abe/test']