import cv2
import numpy as np
import os
import csv
import tensorflow as tf
import multiprocessing


def cut_image(image, x, y, height, width):
    x = int(x)
    y = int(y)
    height = int(height)
    width = int(width)
    return image[y:y+width, x:x+height]

def preprocess_image(image):
    with tf.device("cpu"):
        image = tf.image.resize(image, [32, 32])
        
        image /= 255.0
    return image

class Character:

    def __init__(self, whole_image, name, x, y, height, width):
        image = cut_image(whole_image, x, y, height, width)
        self._image = image
        self._x = x
        self._y = y
        self._height = height
        self._width = width
        self._name = name
        
    @property
    def image(self):
        return self._image
        
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

class Entry:
    
    def __init__(self, image, characters, filename):
        self._image = image
        self._characters = characters
        self._filename = filename

    def __iter__(self):
        return self._characters.__iter__()
        
    @property
    def characters(self):
        return self._characters
        
    @property
    def image(self):
        return self._image
        
    @property
    def filename(self):
        return self._filename

class DataReader:
    
    """
    Usage Examples:
        Construct:
            D = DataReader("Data/train_images", "Data/train.csv")
        Use case 1 recognition (train):
            for d in D.data:
                for c in d.characters:
                    input_ = c.image
                    output = c.name
                    model.fit(input_, output)
        Use case 2 cutting (train):
            for d in D.data:
                input_ = d.image
                output = [(c.x, c.y, c.height, c.width) for c in d.characters]
                model.fit(input_, output)
        Use case 3 cutting and recognition (test):
                for d in D.data:
                    input_ = d.image
                    coordinate = model_cut.eval(input_)
                    for c in coordinate:
                        character_img = cut_image(d.image, *c)
                        character = model_rec.eval(character_img)
    """
    def __init__(self, image_path, label_path, start_index=None, end_index=None):

        with open(label_path) as fp:
            content = csv.reader(fp)
            content = [c for c in content]
        
        self._data = []
        s = max(start_index, 1) if start_index is not None else 1
        e = min(end_index, len(content)) if end_index is not None else len(content)
        if s > len(content):
            return
        for i in range(s, e):
            character_data = content[i][1].split(" ")
            image = cv2.imread(os.path.join(image_path, content[i][0]+".jpg"))
            characters = []
            for j in range(len(character_data)//5):
                characters.append(Character(image, *character_data[j*5:j*5+5]))
            self._data.append(Entry(image, characters, content[i][0]+".jpg"))
            print("Data processing progress: {0}/{1}".format(i, len(content)-1))

    def write_tfrecord(self, file_name):
        all_char = []
        for d in self.data:
            for c in d.characters:
                all_char.append(c)
        all_image = [c.image for c in all_char]
        all_name = [c.name for c in all_char]

        x = list(range(len(all_image)))
        x_ds = path_ds = tf.data.Dataset.from_tensor_slices(x)
         
        char_dict = [x[0] for x in Dictionary("./data/unicode_translation.csv")]
        resized_image = tf.nest.map_structure(preprocess_image, all_image) 
        resized_image = [i.numpy() for i in resized_image]

        labels = [char_dict.index(name) for name in all_name]

        def _float_feature(value):
            """Returns a float_list from a float / double."""
            return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def serialize_example(feature0, feature1):
            feature = {
              'feature0': _int64_feature(feature0),
              'feature1': _float_feature(feature1),
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            return example_proto.SerializeToString()
        
        with tf.io.TFRecordWriter(file_name) as writer:
            for i in range(len(labels)):
                example = serialize_example(labels[i], resized_image[i])
                writer.write(example)

        return resized_image, labels

    def get_dataset(self):
        all_char = []
        for d in self.data:
            for c in d.characters:
                all_char.append(c)
        all_image = [c.image for c in all_char]
        all_name = [c.name for c in all_char]

        x = list(range(len(all_image)))
        x_ds = path_ds = tf.data.Dataset.from_tensor_slices(x)
        
        char_dict = [x[0] for x in Dictionary("./data/unicode_translation.csv")]

        resized_image = tf.nest.map_structure(preprocess_image, all_image) 
 
        image_ds = tf.data.Dataset.from_tensor_slices(tf.cast(resized_image, tf.float64))
        labels = [char_dict.index(name) for name in all_name]
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))

        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds)) 
        ds = image_label_ds.shuffle(buffer_size=len(all_image))
        ds = ds.repeat()
        ds = ds.batch(128)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds
        
    @property
    def data(self):
        return self._data

    def __iter__(self):
        return self._data.__iter__()
            
class Dictionary:

    def __init__(self, file_path):
        data = []
        with open(file_path, "r") as fp:
            content = fp.readlines()
        content = [c.strip() for c in content]
        for d in content:
           data.append(d.split(","))

        self._data = data[1:]

    @property
    def data(self):
        return self._data 

    @property
    def length(self):
        return len(self.data)

    def __iter__(self):
        return self._data.__iter__()
            
