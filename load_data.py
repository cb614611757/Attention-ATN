import tensorflow as tf
import inception_preprocessing
import os
from param import labels_file, train_img_num, test_img_num, num_classes, items_to_descriptions
slim = tf.contrib.slim

labels = open(labels_file, 'r')

labels_to_name = {}
for line in labels:
    label, string_name = line.split(':')
    string_name = string_name[:-1] 
    labels_to_name[int(label)] = string_name

def get_split(split_name, dataset_dir, file_pattern=None, file_pattern_for_counting='IJCAI'):

    if split_name not in ['train', 'validation']:
        raise ValueError('The split_name %s is not recognized. Please input either train or validation as the split_name' % (split_name))
    # pdb.set_trace()
    #Create the full path for a general file_pattern to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    #Count the total number of examples in all of these shard
    num_samples = 0
    file_pattern_for_counting = file_pattern_for_counting + '_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    if split_name == 'train':
        num_samples = train_img_num
    else:
        num_samples = test_img_num

    reader = tf.TFRecordReader

    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
    'image': slim.tfexample_decoder.Image(),
    'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }


    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)


    labels_to_name_dict = labels_to_name


    dataset = slim.dataset.Dataset(
        data_sources = file_pattern_path,
        decoder = decoder,
        reader = reader,
        num_readers = 1,
        num_samples = num_samples,
        num_classes = num_classes,
        labels_to_name = labels_to_name_dict,
        items_to_descriptions = items_to_descriptions)

    return dataset


def load_batch(dataset, batch_size, height=0, width=0, is_training=True):

    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity = 24 + 3 * batch_size,
        common_queue_min = 24)

    raw_image, label = data_provider.get(['image', 'label'])
    image = inception_preprocessing.preprocess_image(raw_image, 224, 224, is_training)

    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    images, raw_images, labels = tf.train.batch(
        [image, raw_image, label],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 4 * batch_size,
        allow_smaller_final_batch = True)

    return images, raw_images, labels
