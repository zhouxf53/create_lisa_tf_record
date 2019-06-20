from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import os
import csv
import numpy as np
import PIL.Image as pil
import tensorflow as tf
import random
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from object_detection.utils.np_box_ops import iou


def convert_lisa_to_tfrecords(data_dir, output_path, label_map_path, validation_set_size):
    """
    
    :param data_dir:
    :param output_path:
    :param label_map_path:
    :param validation_set_size:
    :return:
    """
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    train_count = 0
    val_count = 0

    annotations_dir = os.path.join(data_dir, 'allAnnotations.csv')
    train_writer = tf.python_io.TFRecordWriter(os.path.join(output_path, 'lisa_train.tfrecord'))
    val_writer = tf.python_io.TFRecordWriter(os.path.join(output_path, 'lisa_val.tfrecord'))

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

    :param image_path:
    :param annotations:
    :param label_map_dict:
    :return:
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
        'image/key/sha256': dataset_util.bytes_feature(img_filename.encode('utf8')),
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
    convert_lisa_to_tfrecords(r"D:\DNN_project\traffic_signs_related\signDatabasePublicFramesOnly", r"./data",
                              "./lisa_label_map.pbtxt",
                              500)


if __name__ == '__main__':
    main()
