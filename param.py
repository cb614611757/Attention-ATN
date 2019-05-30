#---------------dataset information-------------#
dataset_dir = 'data/'
labels_file = 'data/my_labels.txt'
file_pattern = 'IJCAI_%s_*.tfrecord'
image_size = 224
num_classes = 110
test_img_num = 0
train_img_num = 1000000
items_to_descriptions = {
    'image': 'A 3-channel RGB coloured flower image that is either tulips, sunflowers, roses, dandelion, or daisy.',
    'label': 'A label that is as such -- 0:daisy, 1:dandelion, 2:roses, 3:sunflowers, 4:tulips'
}


#----------------log and checkpoint------------#
log_dir = './log'

vgg_checkpoint_path = "/home/yq/Downloads/IJCAI/checkpoints/vgg_16/vgg_16.ckpt"
inception_checkpoint_path = "/home/yq/Downloads/IJCAI/checkpoints/inception_v1/inception_v1.ckpt"
resnet__checkpoint_path = "/home/yq/Downloads/IJCAI/checkpoints/resnet_v1_50/model.ckpt-49800"

#--------------------------------------------#
batch_size = 16
batch_shape = [batch_size * 2, image_size, image_size, 3]

max_epsilon = 16.0
eps = 2.0 * max_epsilon / 255.0

max_steps = 50000


