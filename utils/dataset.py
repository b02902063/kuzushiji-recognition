import tensorflow as tf
import csv
import os
import numpy as np


AUTOTUNE = tf.data.experimental.AUTOTUNE

class Character:
        
    def __init__(self, name, x, y, height, width):
        self._x = x
        self._y = y
        self._height = height
        self._width = width
        self._name = name

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def name(self):
        return self._name

def get_dataset(dirpath, csvpath, padded_width, padded_height, batch_size, max_num_character):
       
    all_paths = [os.path.abspath(os.path.join(dirpath, p)) for p in os.listdir(dirpath)]
    path_ds = tf.data.Dataset.from_tensor_slices(all_paths)

    def preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, dtype=tf.float32)
        image = tf.image.pad_to_bounding_box(image, 0, 0, padded_width, padded_height)
        image /= 255.0  # normalize to [0,1] range

        return image

    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        return preprocess_image(image)
   
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    with open(csvpath) as fp:
        content = csv.reader(fp)
        content = [c for c in content]

    all_label_data = dict()
    for i in range(1, len(content)):   
        image_name = content[i][0]
        character_data = content[i][1].split(" ")
        this_character_data = []
        for j in range(len(character_data)//5):
            x = int(character_data[j*5+1])
            y = int(character_data[j*5+2])
            width = int(character_data[j*5+4])
            height = int(character_data[j*5+3])
            this_character_data.append([x, y, width, height])
        try:
            assert(len(this_character_data) <= max_num_character)
        except Exception as e:
            print(len(this_character_data))
            raise e
        while len(this_character_data) < max_num_character:
            this_character_data.append([0, 0, 0, 0])
        all_label_data[image_name] = this_character_data

    labels = []
    for filename in os.listdir(dirpath):
        labels.append(all_label_data[os.path.splitext(filename)[0]])

    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))

    ds = tf.data.Dataset.zip((image_ds, label_ds))
    ds = ds.shuffle(buffer_size=5)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
    
     
    
        
