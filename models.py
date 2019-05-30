import tensorflow as tf
import ops
import utils
import numpy as np
slim = tf.contrib.slim

class Attention_ATN:
  def __init__(self, scope='Gen_v1', is_training=True, ngf=64, norm='instance', image_size=128):
    self.scope = scope
    self.reuse = False
    self.ngf = ngf
    self.norm = norm
    self.is_training = is_training
    self.image_size = image_size

  def __call__(self, input):
    with tf.variable_scope(self.scope):
        with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None,
                            stride=2, weights_initializer=tf.contrib.layers.xavier_initializer()):
            with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                activation_fn=tf.nn.relu, is_training=self.is_training):
              # conv layers
              net = slim.conv2d(input, self.ngf, [3, 3], stride=2, scope='c7s1_32')
              net = slim.batch_norm(net, scope='db1')
              net = slim.conv2d(net, 2*self.ngf, [3, 3], stride=2, scope='d64')
              net = slim.batch_norm(net, scope='db2')
              net = slim.conv2d(net, 4*self.ngf, [3, 3], stride=2, scope='d128')
              net = slim.batch_norm(net, scope='db3')
              net = slim.conv2d(net, 8*self.ngf, [3, 3], stride=2, scope='d256')
              net = slim.batch_norm(net, scope='db4')

              # Resize-convolution layers

              net = tf.image.resize_images(net, (32, 32))
              net = slim.conv2d(net, 4 * self.ngf, [3, 3], stride=1, scope='u128')
              net = slim.batch_norm(net, scope='ub1')
              net = slim.conv2d(net, 4 * self.ngf, [3, 3], stride=1, scope='u128_2')
              net = slim.batch_norm(net, scope='ub1_2')
              net = tf.image.resize_images(net, (64, 64))

              net = slim.conv2d(net, 2 * self.ngf, [3, 3], stride=1, scope='u64')
              net = slim.batch_norm(net, scope='ub2')
              net = slim.conv2d(net, 2 * self.ngf, [3, 3], stride=1, scope='u64_2')
              net = slim.batch_norm(net, scope='ub2_2')
              net = tf.image.resize_images(net, (128, 128))

              net = slim.conv2d(net, 1 * self.ngf, [3, 3], stride=1, scope='u32')
              net = slim.batch_norm(net, scope='ub3')
              net = slim.conv2d(net, 1 * self.ngf, [3, 3], stride=1, scope='u32_2')
              net = slim.batch_norm(net, scope='ub3_2')

              net = tf.image.resize_images(net, (256, 256))
              net = slim.conv2d(net, 0.5 * self.ngf, [3, 3], stride=1, scope='u16')
              net = slim.batch_norm(net, scope='ub4')
              net = slim.conv2d(net, 0.5 * self.ngf, [3, 3], stride=1, scope='u16_2')
              net = slim.batch_norm(net, scope='ub4_2')
              net = tf.image.resize_images(net, (299, 299))

              net = slim.conv2d(net, 0.5 * self.ngf, [3, 3], stride=1, scope='u8')
              net = slim.batch_norm(net, scope='ub5')
              net = slim.conv2d(net, 0.5 * self.ngf, [3, 3], stride=1, scope='u8_2')
              net = slim.batch_norm(net, scope='ub5_2')
              net = slim.conv2d(net, 3, [3, 3], stride=1, scope='u5')
              return net

