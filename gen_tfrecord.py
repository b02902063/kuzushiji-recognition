from utils.data_reader import DataReader
import sys

d = DataReader("./data/test_images", "./data/sample_submission.csv", int(sys.argv[1]), int(sys.argv[2]))
d.write_tfrecord("./TFRecord/test/{0}.tfrecord".format(sys.argv[1]))

