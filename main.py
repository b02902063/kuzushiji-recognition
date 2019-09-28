from image_estimator.recognition import Recognition


recognition_model = Recognition("./TFRecord/train/", "./data/train.csv")   

recognition_model.train()
 
