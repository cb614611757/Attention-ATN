import tensorflow as tf
import os
import logging
import numpy as np
from load_data import get_split, load_batch
from slim.nets import resnet_v1, inception, vgg, inception_resnet_v2, mobilenet_v2
from slim.self_net import resnet_v2, inception_v3, inception_v4
from models import Attention_ATN
from PIL import Image
from cycle_generator import Generator
from utils import save_images
import utils
from reader import Reader
from scipy.misc import imread, imresize
import csv
import ops
import cv2
import pdb
slim = tf.contrib.slim

tf.flags.DEFINE_integer('batch_size', 4, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 299, 'image size, default: 299')
tf.flags.DEFINE_string('input_dir', 'data/',  'The file dir to read imagesdefault:data')
tf.flags.DEFINE_string('output_dir', 'att_atn_images', 'The file dir to save images')
tf.flags.DEFINE_bool('is_training', True, 'Decide train Generator network or not.')
tf.flags.DEFINE_string('checkpoint_path', 'save_models/model.ckpt-116000', 'Path to load checkpoint for Generator network.')
tf.flags.DEFINE_string('save_checkpoint_path', 'att_atn_logs', 'the Path to save checkpoint for gan network.')
tf.flags.DEFINE_integer('num_classes', 110, 'Number of Classes')
FLAGS = tf.flags.FLAGS

CHECKPOINTS_DIR = 'models/'
model_checkpoint_map = {
    'cycle_gan': os.path.join(CHECKPOINTS_DIR, 'cycle_model', 'model.ckpt-190000'),
    'inception_v1': os.path.join(CHECKPOINTS_DIR, 'inception_v1', 'inception_v1.ckpt'),
    'resnet_v1_150': os.path.join(CHECKPOINTS_DIR, 'resnet_v1_50', 'model.ckpt-49800'),
    'vgg_16': os.path.join(CHECKPOINTS_DIR, 'vgg_16', 'vgg_16.ckpt'),
    'inception1_multi': os.path.join(CHECKPOINTS_DIR, 'inception_v1', 'model_multi.ckpt-690461'),
    'resnet_v2_multi': os.path.join(CHECKPOINTS_DIR, 'resnet_v2_152', 'model_multi.ckpt-917617'),
    'inception_resnet_v2': os.path.join(CHECKPOINTS_DIR, 'inception_resnet_v2', 'model.ckpt-231498'),
    'mobilenet_v2': os.path.join(CHECKPOINTS_DIR, 'mobilenet_v2', 'model.ckpt-279921'),
    'inception_v3': os.path.join(CHECKPOINTS_DIR, 'inception_v3', 'model.ckpt-586654'),
    'inception_v4': os.path.join(CHECKPOINTS_DIR, 'inception_v4', 'model.ckpt-384623')}

# note: if you use another version tensorflow, the name of logit and conv layer may be different
logit_name_list = ["Logits", "resnet_v1_50/logits", "vgg_16/fc8"]
conv_name_list = ["Mixed_5c", "resnet_v1_50/block4/unit_3/bottleneck_v1", "vgg_16/pool5"]
num_model = 3

# load input images to generate attack images
def load_input_images(input_dir, batch_shape):
    filenames = []
    input_images = np.zeros(shape=[FLAGS.batch_size, 224, 224,3])
    raw_images = np.zeros(batch_shape)
    trueLabels = np.zeros(batch_shape[0], dtype=np.int32)
    idx = 0
    batch_size = batch_shape[0]
    with open(os.path.join(input_dir, 'dev.csv'), 'rb') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = os.path.join(input_dir, row['filename'])
            with open(filepath) as f:
                raw_image = imread(f, mode='RGB').astype(np.float)
                raw_image = (raw_image / 255.0) * 2.0 - 1.0
                image = imresize(raw_image, [224, 224, 3])
            input_images[idx, :, :, :] = image
            raw_images[idx, :, :, :] = raw_image
            trueLabels[idx] = int(row['trueLabel'])
            filenames.append(os.path.basename(filepath))
            idx += 1
            if idx == batch_size:
                yield filenames, input_images, raw_images, trueLabels
                filenames = []
                input_images = np.zeros(shape=[FLAGS.batch_size, 224, 224, 3])
                raw_images = np.zeros(batch_shape)
                trueLabels = np.zeros(batch_shape[0], dtype=np.int32)
                idx = 0
        if idx > 0:
            yield filenames, input_images, raw_images, trueLabels


def preprocess_for_model(images, model_type):
    if 'inception' in model_type.lower():
        images = tf.image.resize_bilinear(images, [224, 224], align_corners=False)
        # tensor-scalar operation
        images = (images / 255.0) * 2.0 - 1.0
        return images

    if 'inceptiion_reset' in model_type.lower():
        images = tf.image.resize_bilinear(images, [299, 299], align_corners=False)
        return images

    if 'mobilenet' in model_type.lower():
        images = tf.image.resize_bilinear(images, [224, 224], align_corners=False)
        return images

    if 'resnet' in model_type.lower() or 'vgg' in model_type.lower():
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        images = tf.image.resize_bilinear(images, [224, 224], align_corners=False)
        tmp_0 = images[:, :, :, 0] - _R_MEAN
        tmp_1 = images[:, :, :, 1] - _G_MEAN
        tmp_2 = images[:, :, :, 2] - _B_MEAN
        images = tf.stack([tmp_0, tmp_1, tmp_2], 3)
        return images

def get_predict_from_model(X, num_classes, reuse = False):
    # model 1: InceptionV1
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits_inc, end_points_inc_v1 = inception.inception_v1(
            X, num_classes=num_classes, is_training=False, reuse=reuse, scope='InceptionV1')

    # model 4_1: resnet_v1_150
    image = (((X + 1.0) * 0.5) * 255.0)
    processed_imgs_res_v2_152 = preprocess_for_model(image, 'resnet_v2_152')
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits_res_v1_150, end_points_res_v1_150 = resnet_v1.resnet_v1_50(
            processed_imgs_res_v2_152, num_classes=num_classes,  is_training=False, reuse=reuse, scope='resnet_v1_50')

    # model 5: vgg_16
    processed_imgs_vgg_16 = preprocess_for_model(image, 'vgg_16')
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits_vgg_16, end_points_vgg_16 = vgg.vgg_16(
            processed_imgs_vgg_16, num_classes=num_classes, is_training=False, reuse=reuse, scope='vgg_16')

    return [end_points_inc_v1, end_points_res_v1_150, end_points_vgg_16]

def Discriminater(X, num_classes, reuse = False):

    # model 1: InceptionV1
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits_inc, end_points_inc_v1 = inception.inception_v1(
            X, num_classes=num_classes, is_training=False, reuse=reuse, scope='Inception_multi')
    # model 2: inceptiion_reset
    processed_imgs_ince_res_v1_50 = preprocess_for_model(X, 'inception_reset')
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_inc_res_v2, end_points_inc_res_v2 = inception_resnet_v2.inception_resnet_v2(
            processed_imgs_ince_res_v1_50, num_classes=num_classes, is_training=False, reuse=reuse,
            scope='InceptionResnetV2')
    # model 3: mobilenet
    processed_imgs_mobile_v2 = X
    with slim.arg_scope(mobilenet_v2.training_scope()):
        logits_mobile_v2, end_points_mobile_v2 = mobilenet_v2.mobilenet_v2_140(
            processed_imgs_mobile_v2, num_classes=num_classes, is_training=False, reuse=reuse, scope='MobilenetV2')

    # model 4_2: resnet_v2_152
    image = (((X + 1.0) * 0.5) * 255.0)
    processed_imgs_res_v2_152 = preprocess_for_model(image, 'resnet_v2_152')
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_res_v2_152, end_points_res_v2_152 = resnet_v2.resnet_v2_152(
            processed_imgs_res_v2_152, num_classes=num_classes, is_training=False, reuse=reuse, scope='resnet_multi')

    # model 1_3: inceptiion_v3_multi
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_inception_v3, _ = inception_v3.inception_v3(
            processed_imgs_ince_res_v1_50, num_classes=num_classes, is_training=False, reuse=reuse, scope='InceptionV3')

    # model 1_7: inceptiion_v4_multi
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_inception_v4, _ = inception_v4.inception_v4(
            processed_imgs_ince_res_v1_50, num_classes=num_classes, is_training=False, reuse=reuse, scope='InceptionV4')

    return [tf.squeeze(logits_inc), tf.squeeze(logits_res_v2_152), tf.squeeze(logits_inc_res_v2), \
            tf.squeeze(logits_mobile_v2), tf.squeeze(logits_inception_v3), tf.squeeze(logits_inception_v4)]

# get the Grad-CAM regions for a tzrget conv layer
def get_cam(grad_tensor, conv_tensor):
    weights = tf.reduce_mean(grad_tensor, axis=(1, 2), keep_dims=True)
    cam_tensor = conv_tensor * weights
    cam_tensor = tf.image.resize_bilinear(cam_tensor, [299, 299])
    cam_tensor = tf.reduce_sum(cam_tensor, axis=3)
    cam_tensor = tf.abs(cam_tensor)
    cam_max = tf.reduce_max(cam_tensor, axis=(1, 2), keep_dims=True)
    cam_tensor = cam_tensor / cam_max   # 1*299*299
    return tf.expand_dims(cam_tensor, axis=3)

#    calculate the synthesis image and cam_matrix
def cal_synthesis_image_and_cam_matrix(x_input, raw_input, labels, num_classes, logit_name_list, conv_names_list, delt_d):
    input_to_model = tf.image.resize_bilinear(raw_input, [224, 224], align_corners=False)
    end_points = get_predict_from_model(input_to_model, num_classes, reuse=False)
    model_cam_matrix_sum = 0
    for index in range(3):
        end_point = end_points[index][logit_name_list[index]]
        y_c = tf.reduce_sum(tf.multiply(end_point, tf.one_hot(labels, num_classes)), axis=1)
        target_conv_layer = end_points[index][conv_names_list[index]]
        target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]
        model_cam_matrix = get_cam(target_conv_layer_grad, target_conv_layer)
        model_cam_matrix_sum += model_cam_matrix
    model_cam_matrix_average = tf.divide(model_cam_matrix_sum, num_model)

    one = tf.ones_like(model_cam_matrix_average)
    zero = tf.zeros_like(model_cam_matrix_average)
    model_cam_matrix_ave = tf.where(model_cam_matrix_average < delt_d, x=zero, y=one)

    x_output = (tf.subtract(1.0, model_cam_matrix_ave, name=None)) * raw_input + model_cam_matrix_ave * x_input
    x_output = tf.clip_by_value(x_output, -1, 1)
    return x_output, model_cam_matrix_average

# loss function to encourage misclassification after perturbation
def get_cross_entropy(labels, preds):
  one_hot_labels = tf.one_hot(labels, FLAGS.num_classes)
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot_labels, preds)
  cross_entropy = tf.clip_by_value(cross_entropy, 0.0, 10.0)
  return cross_entropy

def adv_loss(preds_list, labels):
    attack_loss_list = list()
    for i in range(len(preds_list)):
        attack_loss = get_cross_entropy(labels, preds_list[i])
        attack_loss_list.append(attack_loss)
    return tf.reduce_sum(tf.convert_to_tensor(attack_loss_list))

def perturb_loss(input_image, gan_image):
    X = (((input_image + 1.0) * 0.5) * 255.0)
    Y = (((gan_image + 1.0) * 0.5) * 255.0)

    X = tf.reshape(X, shape=[-1, 3])
    Y = tf.reshape(Y, shape=[-1, 3])
    dis_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum((X - Y) ** 2, axis=1)))
    # dis_loss = tf.sqrt(tf.reduce_mean(tf.square(X - Y)))
    return dis_loss

def test_accuracy(labels, preds):
    correct = tf.nn.in_top_k(preds, labels, 1)
    correct = tf.cast(correct, tf.float16)
    accuracy = tf.reduce_mean(correct)
    return accuracy

def test_all_accaracy(labels, preds_list):
    attack_acc_list = list()
    for i in range(len(preds_list)):
        acc = test_accuracy(labels, preds_list[i])
        attack_acc_list.append(acc)
    return tf.reduce_mean(tf.convert_to_tensor(attack_acc_list))

def image_show(image):
    image = (((image + 1.0) * 0.5) * 255.0).astype(np.uint8)
    image = imresize(image, [299, 299])
    Image.fromarray(image).show()

def train():
    graph = tf.Graph()
    with graph.as_default():
	# input the data with TensorFlow's native TFRecord format.
        train_dataset = get_split('train', FLAGS.input_dir, file_pattern='IJCAI_%s_*.tfrecord')
        train_images, raw_images, train_labels = load_batch(train_dataset, batch_size=FLAGS.batch_size, height=299,
                                                            width=299)
        raw_images_cast = tf.cast(raw_images, tf.float32)
        scale_raw_image = raw_images_cast / 127.5 - 1.0
 
        G_self = Attention_ATN(scope='Generate', is_training=FLAGS.is_training, ngf=64, norm='instance', image_size=FLAGS.image_size)
        fake_images_self = G_self(scale_raw_image)

        G = Generator('F', is_training=False, ngf=64, norm='instance', image_size=FLAGS.image_size)
        x_adv = fake_images_self
        fake_images_be_saved = (tf.cast(x_adv, tf.float32)+1.0) * 127.5

        attack_image, cam_matrix_show = cal_synthesis_image_and_cam_matrix(x_adv, scale_raw_image, train_labels, 110, \
                                                       logit_name_list, conv_name_list, 0.2)

        # transform the image(attack_image and raw image) to a unifield space through the cycle-gan 
        attack_gan_image = G(tf.reshape(attack_image, [FLAGS.batch_size, 299, 299, 3]))
        raw_gan_image = G(tf.reshape(scale_raw_image, [FLAGS.batch_size, 299, 299, 3]))

        target_input_perturb = tf.image.resize_bilinear(attack_gan_image, [224, 224], align_corners=False)
        raw_gan_image = tf.image.resize_bilinear(raw_gan_image, [224, 224], align_corners=False)

        fake_logit_list = Discriminater(X=target_input_perturb, num_classes=110, reuse=False)
        true_logit_list = Discriminater(X=raw_gan_image, num_classes=110, reuse=True)

        #  loss to test the pixel distence
        l_perturb = perturb_loss(scale_raw_image, x_adv)
        #  loss to test the atteck efficiency
        l_adv = adv_loss(fake_logit_list, train_labels)
        true_adv = adv_loss(true_logit_list, train_labels)

        beta = 1
        gamma = 1
	
	# total loss ,alpha and beta are used to control the relative importance of the each loss. 
        g_loss = beta * l_perturb - gamma * l_adv

        true_accaracy = test_all_accaracy(train_labels, true_logit_list)
        fake_accaracy = test_all_accaracy(train_labels, fake_logit_list)

        ge_var_list = slim.get_model_variables(scope='Generate')
        g_opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(g_loss, var_list=ge_var_list)

        tf.summary.scalar('total_loss', g_loss)
        tf.summary.scalar('perturb_loss', l_perturb)
        tf.summary.scalar('adv_loss', l_adv)
        tf.summary.scalar('fake_image_accaracy', fake_accaracy)
        tf.summary.image('Attack_image', attack_image)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.save_checkpoint_path, graph)

        saver = tf.train.Saver(var_list=ge_var_list, max_to_keep=6)
        step = 0

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        # model used to calculate the Grad_CAM 
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))

	# defense model
        variables_to_restore_v4 = slim.get_variables_to_restore(include=['InceptionResnetV2'], exclude=['InceptionResnetV2/AuxLogits'])
        s4 = tf.train.Saver(variables_to_restore_v4)
        s5 = tf.train.Saver(slim.get_model_variables(scope='MobilenetV2'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='Inception_multi'))
        s7 = tf.train.Saver(slim.get_model_variables(scope='resnet_multi'))
        variables_to_restore_v3 = slim.get_variables_to_restore(include=['InceptionV3'], exclude=['InceptionV3/AuxLogits'])
        s8 = tf.train.Saver(variables_to_restore_v3)

        variables_to_restore_v4 = slim.get_variables_to_restore(include=['InceptionV4'], exclude=['InceptionV4/AuxLogits'])
        s9 = tf.train.Saver(variables_to_restore_v4)

        s1.restore(sess, model_checkpoint_map['inception_v1'])
        s2.restore(sess, model_checkpoint_map['resnet_v1_150'])
        s3.restore(sess, model_checkpoint_map['vgg_16'])

        s4.restore(sess, model_checkpoint_map['inception_resnet_v2'])
        s5.restore(sess, model_checkpoint_map['mobilenet_v2'])
        s6.restore(sess, model_checkpoint_map['inception1_multi'])
        s7.restore(sess, model_checkpoint_map['resnet_v2_multi'])
        s8.restore(sess, model_checkpoint_map['inception_v3'])
        s9.restore(sess, model_checkpoint_map['inception_v4'])

	# Attention_ATN model, if you want to train a model from scratch, commit the below codes
        #s = tf.train.Saver(tf.global_variables(scope='Generate'))
        #s.restore(sess, FLAGS.checkpoint_path)

	# cycle gan model
        s_gan = tf.train.Saver([var for var in tf.global_variables() if 'F' in var.name])
        s_gan.restore(sess, model_checkpoint_map['cycle_gan'])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        while step < 200001:
            true_acc, attack_image_value, cam_matrix_value, fake_acc, _, loss0, loss1, loss2, fake_image, G_gan_loss_val, summary = (sess.run([true_accaracy, \
            attack_image, cam_matrix_show, fake_accaracy, g_opt, true_adv, l_adv, l_perturb, fake_images_be_saved, g_loss, summary_op]))

            if step % 10 == 0:
              logging.info('-----------Step %d:-------------' % step)
              logging.info('  real_image_accuracy  : {}'.format(true_acc))
              logging.info('  fake_image_accuracy  : {}'.format(fake_acc))
              logging.info('  real_image_loss  : {}'.format(loss0))
              logging.info('  Discriminator_loss  : {}'.format(loss1))
              logging.info('  Pix_distance_loss   : {}'.format(loss2))
              logging.info('  G_loss   : {}'.format(G_gan_loss_val))

              train_writer.add_summary(summary, step)
              train_writer.flush()

            if step > 5000 and step % 500 == 0:
              save_path = saver.save(sess, FLAGS.save_checkpoint_path + "/model.ckpt", global_step=step)
              logging.info("Model saved in file: %s" % save_path)
              for index in range(FLAGS.batch_size):
                save_images(fake_image[index], [299, 299, 3], FLAGS.output_dir+'/'+str(step)+str(index)+'.png')

            step += 1

        save_path = saver.save(sess, FLAGS.save_checkpoint_path + "/model.ckpt", global_step=step)
        logging.info("Model saved in file: %s" % save_path)

        # When done, ask the threads to stop.
        coord.request_stop()
        coord.join(threads)

def main(unused_argv):
  train()

if __name__ == '__main__':
  os.environ["CUDA_VISIBLE_DEVICES"] = '2'
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.Session(config=config)
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
