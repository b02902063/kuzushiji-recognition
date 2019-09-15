import cv2
import numpy as np
import os
import csv


def cut_image(image, x, y ,height, width):
    return image[y:y+width, x:x+height]

class Character:

    def __init__(self, name, whole_image, x, y, height, width):
        self._image = cut_image(whole_image, x, y, height, width)
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
    
    def __init__(self, image, characters, fileanme):
        self._image = image
        self._characters = characters
        self._filename = filename
        
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
    Usage Example:
        Construct:
            D = DataReader("Data/train_images", "Data/train.csv")
        Use case 1 recognition (train):
            for d in D.data:
                for c in d.characters:
                    input_ = c.image
                    output = c.name
                    model.fit(input_, output)
        Use case 2 cutting (train):
            for d in D,data:
                input_ = d.image
                output = [(c.x, c.y, c.height, c.width) for c in d.characters]
                model.fit(input_, output)
        Use case 3 cutting and recognition (test):
                for d in D,data:
                    input_ = d.image
                    coordinate = model_cut.eval(input_)
                    for c in coordinate:
                        character_img = cut_image(d.image, *c)
                        character = model_rec.eval(character_img)
    """
    def __init__(self, image_path, label_path):

        with open(label_path) as fp:
            content = csv.reader(fp)
            content = [c for c in content]
        
        self._data = []
        for i in range(1, len(content)):
            character_data = content[i][1].split(" ")
            image = cv2.imread(os.path.join(image_path, character_data[i][0]+".jpg"))
            characters = []
            for j in range(len(character_data)//5):
                characters.append(Character(image, *character_data[j*5:character_data[j*5]+5]))
            self._data.append(Entry(image, characters, character_data[i][0]))
        
    @property
    def data:
        return self._data
            
            
            