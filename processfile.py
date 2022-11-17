import os
import cv2
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

path_csv = r"C:\Users\doant\Dropbox\Dataset Label.csv"
path_fruit_gray = r"C:\Users\doant\Dropbox\folder"

image_matrix = []
for i in os.listdir(path_fruit_gray):
    j = os.path.join(path_fruit_gray, i)
    for fruit in os.listdir(j):
        image = os.path.join(j, fruit)
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_matrix.append(img)

image_matrix = np.array(image_matrix)
image_matrix = image_matrix/255

target = pd.read_csv(path_csv)
label = target.iloc[:,-1]
Label = LabelEncoder()
y = Label.fit_transform(label)
y = np_utils.to_categorical(y,131)

np.save("./datatrain/label",y)
np.save("./datatrain/data", image_matrix)















