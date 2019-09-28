import tensorflow as tf
import os


def get_dataset(filenames, batch_size=32, repeat=True):
    dataset = tf.data.TFRecordDataset(filenames)

    image_feature_description = {
        'feature0': tf.io.FixedLenFeature([], tf.int64),
        'feature1': tf.io.FixedLenFeature([32, 32, 3], tf.float32),
    }
    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        a = list(tf.io.parse_single_example(example_proto, image_feature_description).values())
        return a[1], a[0]
    dataset = dataset.map(_parse_image_function)
    dataset = dataset.shuffle(buffer_size=100).batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    return dataset 

def get_train_and_validation_dataset(filepath, ratio=0.9, batch_size=32):

    filenames = [os.path.join(filepath, a) for a in os.listdir(filepath)]
    train_len = int(len(filenames) * 0.9)
    train = filenames[:train_len]
    valid = filenames[train_len:]

    return get_dataset(train, batch_size=batch_size, repeat=True), get_dataset(valid, batch_size=batch_size, repeat=False)
