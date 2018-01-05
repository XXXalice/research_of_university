import sys
import os
import cv2
import numpy as np
import tensorflow as tf

import hyperParams as hp

def inference(images_placeholder, keep_prob):
    def weight_variables(shape):
        init = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init)

    def bias_variables(shape):
        init = tf.constant(0.1, shape=shape)
        return tf.Variable(init)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W , strides=[1,1,1,1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    x_image = tf.reshape(images_placeholder, [-1,28,28,3])

    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variables([5,5,3,16])
        b_conv1 = bias_variables([16])
        h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
        tf.summary.histogram("wc1", W_conv1)

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variables([5,5,16,32])
        b_conv2 = bias_variables([32])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        tf.summary.histogram('wc2', W_conv2)

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3') as scope:
        W_conv3 = weight_variables([5,5,32,64])
        b_conv3 = bias_variables([64])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        tf.summary.histogram('wc2', W_conv3)

    with tf.name_scope('pool3') as scope:
        h_pool3 = max_pool_2x2(h_conv3)

    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variables([4*4*64, 1024])
        b_fc1 = bias_variables([1024])
        h_pool3_flat = tf.reshape(h_pool3,[-1, 4*4*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variables([1024, hp.NUM_CLASSES])
        b_fc2 = bias_variables([hp.NUM_CLASSES])

    with tf.name_scope('softmax') as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv

def loss(logits, labels):
    cross_entropy = -tf.reduce_sum(labels * tf.log(logits))
    tf.summary.scalar('cross_entropy', cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1),tf.argmax(labels, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    tf.summary.scalar('accuracy',acc)
    return acc


def make_asarray():
    f = open(hp.FLAGS.train, 'r')
    train_image = []
    train_label = []
    for line in f:
        line = line.rstrip()
        l = line.split()
        img = cv2.imread(hp.FLAGS.img_dir + '/train/' + l[0])
        img = cv2.resize(img, (28, 28))
        train_image.append(img.flatten().astype(np.float32) / 255.0)
        tmp = np.zeros(hp.NUM_CLASSES)
        tmp[int(l[1])] = 1
        train_label.append(tmp)
    train_image = np.asarray(train_image)
    train_label = np.asarray(train_label)
    f.close()

    f = open(hp.FLAGS.test, 'r')
    test_image = []
    test_label = []
    for line in f:
        line = line.rstrip()
        l = line.split()
        img = cv2.imread(hp.FLAGS.img_dir + '/test/' + l[0])
        img = cv2.resize(img, (28, 28))
        test_image.append(img.flatten().astype(np.float32) / 255.0)
        tmp = np.zeros(hp.NUM_CLASSES)
        tmp[int(l[1])] = 1
        test_label.append(tmp)
    test_image = np.asarray(test_image)
    test_label = np.asarray(test_label)
    f.close()

    return train_image, train_label, test_image, test_label


def main():
    train_image, train_label, test_image, test_label = make_asarray()
    print('画像の数値化に成功しました')
    with tf.Graph().as_default():
        #空のテンソル定義 shapeの第一引数は制限数（Noneで制限なし） 第二引数はサイズ
        images_placeholder = tf.placeholder('float', shape=(None, hp.IMAGE_PIXELS))
        labels_placeholder = tf.placeholder('float', shape=(None, hp.NUM_CLASSES))
        #dropout率を入れるテンソル定義
        keep_prob = tf.placeholder('float')
        #モデル生成
        logits = inference(images_placeholder,keep_prob)
        loss_value = loss(logits, labels_placeholder)
        train_op = training(loss_value, hp.FLAGS.learning_rate)
        acc = accuracy(logits, labels_placeholder)
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(hp.FLAGS.train_dir, sess.graph)

        for steps in range(hp.FLAGS.step_num):
            for i in range(int(len(train_image)/hp.FLAGS.batch_size)):
                batch = hp.FLAGS.batch_size * i
                sess.run(train_op, feed_dict={
                    images_placeholder: train_image[batch:batch+hp.FLAGS.batch_size],
                    labels_placeholder: train_label[batch:batch+hp.FLAGS.batch_size],
                    keep_prob: 0.5
                })

            train_accuracy = sess.run(acc, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0
            })
            print('step'+ str(steps) + ' training accuracy '+ str(train_accuracy))

            summary_str = sess.run(summary_op, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0
            })
            summary_writer.add_summary(summary_str, steps)

        print('-----start test--------')

        test_accuracy = sess.run(acc, feed_dict={
            images_placeholder: test_image,
            labels_placeholder: test_label,
            keep_prob: 1.0
        })
        print('test accuracy ' + str(test_accuracy))

        save_path = saver.save(sess, os.getcwd()+'/model.ckpt')

if __name__ == '__main__':
    main()