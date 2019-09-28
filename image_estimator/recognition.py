from utils.data_reader import DataReader, Dictionary
from utils import tfrecord_dataset
from image_estimator import model
import pickle
import os 
import tensorflow as tf
    

class Recognition:

    def __init__(self, train_images, label_csv):
        self.all_data = Dictionary("./data/unicode_translation.csv")
        self.model = model.resnet_v2([32, 32, 3], 110, self.all_data.length)  
        """
        self.dataset = DataReader(train_images, label_csv)
        with open("./data/data.pickle", "wb+") as fp:
            pickle.dump(self.dataset, fp)
        i = 0
        for d in self.dataset:
            j = 0
            for c in d:
                print(c.image)
                print(c.name)
                j += 1
                if j > 4:
                    break
            i += 1
            if i > 1:
                break
        """

    def train(self):

        train_dataset, valid_dataset = \
            tfrecord_dataset.get_train_and_validation_dataset("./TFRecord/train", batch_size=64)

        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           metrics=['accuracy']) 

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="./model/recognition_model.h5",
                                                        monitor='val_acc',
                                                        verbose=1,
                                                        save_best_only=True)
        
        def lr_schedule(epoch):
            if epoch < 10:
                return 0.001
            else:
                return 0.001 * tf.math.exp(0.1 * (10 - epoch))
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        
        #tensorboard = tf.keras.callbacks.TensorBoard(log_dir="./model")

        callbacks = [checkpoint, lr_scheduler]

        self.model.fit(x=train_dataset, epochs=1000, callbacks=callbacks, validation_data=valid_dataset, use_multiprocessing=True, steps_per_epoch=1000)

