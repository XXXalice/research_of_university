import tensorflow as tf

NUM_CLASSES = 2 #分類数
IMAGE_SIZE = 28 #画像の一片のサイズ
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 3 #画像のピクセル数（縦×横の300%）

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('img_dir', './img', 'Directory of images')
tf.app.flags.DEFINE_string('train_dir', './logs', 'Directory to put the training data')
tf.app.flags.DEFINE_string('train', './data/train.txt', 'train data files')
tf.app.flags.DEFINE_string('test', './data/test.txt', 'test data files')
tf.app.flags.DEFINE_integer('step_num', 300, 'Number of iter to run trainer')
tf.app.flags.DEFINE_integer('batch_size', 8, 'Number of batch for once iterate')
tf.app.flags.DEFINE_float('learning_rate', 1e-5, 'Update rate in loss function of once')

