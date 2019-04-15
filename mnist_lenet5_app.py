#!D:\Software\Anaconda\InstallAddress\envs\tensorflow\python.exe
# -*- coding: utf-8 -*- 
# @Time : 2019/4/14 16:59 
# @Author : Howard 
# @Site :  
# @File : mnist_lenet5_app.py 
# @Software: PyCharm
import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_lenet5_backward
import mnist_lenet5_forward


def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [1, mnist_lenet5_forward.IMAGE_SIZE * mnist_lenet5_forward.IMAGE_SIZE])
        y = mnist_lenet5_forward.forward(x=x, train=False, state=2, regularizer=None)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(mnist_lenet5_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_lenet5_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print('No checkpoint file found')
                return -1


def pre_pic(img):
    img = Image.fromarray(img)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    threshold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if im_arr[i][j] < threshold:
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)

    return img_ready


def application(img):
    img_Arr = pre_pic(img)
    preValue = restore_model(img_Arr)
    return preValue
