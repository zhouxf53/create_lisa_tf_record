### Converts lisa detection dataset to TFRecords with a standard format allowing to use this dataset to train tensorflow object detectors.
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
  
Example usage
```asciidoc
    python create_lisa_tf_record.py \
          --data_dir=../signDatabasePublicFramesOnly \
          --output_path=./data \
          --label_map_path=./lisa_label_map.pbtxt
```
