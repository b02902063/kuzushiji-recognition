import cv2
import numpy as np
import os
import csv


class Entry:
    
    def __init__(self, image, label, fileanme):
        self._image = image
        self._label = label
        self._filename = filename
        
    @property
    def label(self):
        return self._label
        
    @property
    def image(self):
        return self._image
        
    @property
    def filename(self):
        return self._filename

class DataReader:
    
    def __init__(self, image_path, label_path):

        with open(label_path) as fp:
            content = csv.reader(fp)
            content = [c for c in content]
        
        image_dict = dict()
        for i in range(1, len(content)):
            character_data = content[i][1].split(" ")
            character_dict = dict()
            for j in range(len(character_data)//5):
                character_dict[character_data[j*5]] = character_data[j*5+1:j*5+5]
            image_dict[content[i][0]] = character_dict
            
        self._data = []
        for filename in image_dict:
            image = cv2.imread(os.path.join(image_path, filename+".jpg"))
            self._data.append(Entry(image, image_dict[filename],filename))
            
    @property
    def data:
        return self._data
            
            
            