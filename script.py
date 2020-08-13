from tensorflow.keras.models import load_model
import tensorflow
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import os
from matplotlib import pyplot as plt
import numpy as np
import shutil


model = load_model("celeb_tensorboard.h5")

ImageDataGenerator = tensorflow.keras.preprocessing.image.ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)


val_dir =  "5-celebrity-faces-dataset/val"
validation_generator = test_datagen.flow_from_directory(
val_dir,
target_size=(150, 150),
batch_size=2,
class_mode='categorical')

val_dir_list = os.listdir(val_dir)
label_names = dict(zip(range(len(val_dir_list)),val_dir_list))


def label_converter(label):
    label = np.squeeze(label.round()).tolist()
    if 1.0 in label:
        label=label.index(1.0)
    else:
        label = 0    
    return label_names[label]


c=0
images = []
labels = []
for data_batch, labels_batch in validation_generator:
    images.extend(data_batch)
    labels.extend(labels_batch)
    c+=1
    if c==5:
        break

images = np.array(images)
labels = np.array(labels)

if "Predictions" in os.listdir("."): 
    shutil.rmtree("Predictions")
os.mkdir("Predictions")

for folder in val_dir_list:
    os.mkdir("Predictions/"+folder)
    
    
from tensorflow.keras.preprocessing.image import save_img

for image, label, i in zip(images, labels, range(len(labels))):
    prediction = label_converter(model.predict(image.reshape(1,150,150,3)))
    save_img(f"Predictions/{prediction}/{i}.jpg",image)
    
