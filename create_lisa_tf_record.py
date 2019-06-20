""" Convert raw lisa detection dataset to TFRecord for object_detection.

  Converts lisa detection dataset to TFRecords with a standard format allowing
  to use this dataset to train object detectors. The raw dataset can be
  downloaded from:
  http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html

  lisa detection dataset contains 7855 training images. Using this code with
  the default settings will set aside 500 images randomly with as a validation set.
  This can be altered using the flags, see details below.

  Tensorflow object detection API is required to be installed, check:
  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

  Most of the credits of this script go to:
  https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_kitti_tf_record.py
  https://github.com/cooliscool/LISA-on-SSD-mobilenet/blob/master/create_lisa_tfrecords.py

  Example usage:
    python create_lisa_tf_record.py \
        --data_dir=/home/user/lisa \
        --output_path=/home/user/lisa.record
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import os
import csv
import PIL.Image as pil
import tensorflow as tf
import random

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

tf.app.flags.DEFINE_string('data_dir', '', 'Location of root directory for the '
                                           'data. Folder structure is assumed to be named as:'
                                           'signDatabasePublicFramesOnly'
                                           '<data_dir>/allAnnotations.csv,'
                                           '<data_dir>/aiua120214-0/'
                                           '...')
tf.app.flags.DEFINE_string('output_path', '', 'Path to which TFRecord files'
                                              'will be written. The TFRecord with the training set'
                                              'will be located at: <output_path>/LISA_train.tfrecord.'
                                              'And the TFRecord with the validation set will be'
                                              'located at: <output_path>/LISA__val.tfrecord')
tf.app.flags.DEFINE_string('label_map_path', 'data/kitti_label_map.pbtxt',
                           'Path to label map proto.')
tf.app.flags.DEFINE_integer('validation_set_size', '500', 'Number of images to'
                                                          'be used as a validation set.')
FLAGS = tf.app.flags.FLAGS


def convert_lisa_to_tfrecords(data_dir, output_path, label_map_path, validation_set_size):
    """
    Convert the LISA detection dataset to TFRecords.
    :param data_dir: directory with the name "signDatabasePublicFramesOnly"
    :param output_path: suggest ./data
    :param label_map_path: full path to the label_map
    :param validation_set_size: default of 500 with flag settings
    :return: N/A
    """
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    train_count = 0
    val_count = 0

    annotations_dir = os.path.join(data_dir, 'allAnnotations.csv')
    train_writer = tf.python_io.TFRecordWriter(os.path.join(output_path, 'LISA_train.tfrecord'))
    val_writer = tf.python_io.TFRecordWriter(os.path.join(output_path, 'LISA_val.tfrecord'))

    # parse annotation csv file
    with open(annotations_dir) as csvFile:
        data_reader = csv.reader(csvFile, delimiter=';')
        next(data_reader)  # for skipping first row
        parsed_annotations = []
        for row in data_reader:
            parsed_annotations.append([row])

    random.seed(49)
    random.shuffle(parsed_annotations)

    for img_num, parsed_annotation in enumerate(parsed_annotations):
        is_validation_img = img_num < validation_set_size
        image_path = os.path.join(data_dir, parsed_annotation[0][0])
        example = prepare_example(image_path, parsed_annotation[0], label_map_dict)
        if is_validation_img:
            val_writer.write(example.SerializeToString())
            val_count += 1
        else:
            train_writer.write(example.SerializeToString())
            train_count += 1

    train_writer.close()
    val_writer.close()
    print("trained with %s images and validated with %s images" % (train_count, val_count))


def prepare_example(image_path, annotations, label_map_dict):
    """
    Converts a dictionary with annotations for an image to tf.Example proto.
    :param image_path: full path to the image
    :param annotations: a list object obtained by reading the annotation csv file
    :param label_map_dict: a map from string label names to integer ids.
    :return: example: The converted tf.Example.
    """
    print("encoding %s" % image_path)
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    image = pil.open(encoded_png_io)

    if image.format != 'PNG':
        raise ValueError('Image format error')

    key = hashlib.sha256(encoded_png).hexdigest()
    # obtain attributes
    width, height = image.size
    img_filename = image_path.split('/')[-1]

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    occlud = []

    xmin.append(int(annotations[2]) / width)
    ymin.append(int(annotations[3]) / height)
    xmax.append(int(annotations[4]) / width)
    ymax.append(int(annotations[5]) / height)
    class_name = annotations[1]
    classes_text.append(class_name)
    classes.append(label_map_dict[class_name])
    classes_text = [class_text.encode('utf-8') for class_text in classes_text]
    trun, occ = annotations[6].split(',')
    truncated.append(int(trun))
    occlud.append(int(occ))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(img_filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(img_filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_png),
        'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.int64_list_feature(occlud),
    }))
    return example


def main():
    convert_lisa_to_tfrecords(
        data_dir=FLAGS.data_dir,
        output_path=FLAGS.output_path,
        label_map_path=FLAGS.label_map_path,
        validation_set_size=FLAGS.validation_set_size
    )


if __name__ == '__main__':
    tf.app.run()
