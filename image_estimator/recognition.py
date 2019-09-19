from utils.data_reader import DataReader


class Recognition:

    def __init__(self, train_images, label_csv):
        self.dataset = DataReader(train_images, label_csv)
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
        
