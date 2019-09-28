from utils.data_reader import DataReader
import sys

d = DataReader("./data/train_images", "./data/train.csv", int(sys.argv[1]), int(sys.argv[2]))
d.write_tfrecord("./TFRecord/train/{0}.tfrecord".format(sys.argv[1]))

