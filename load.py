import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
import matplotlib.pyplot as plt
import keras

folder = []
for i in os.listdir("datatrain"):
    file = os.path.join("datatrain",i)
    folder.append(file)

x = np.load(folder[0])
y = np.load(folder[1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(100,100,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(131, activation='softmax'))

optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
model.summary()
history = model.fit(x_train, y_train, validation_split=0.1, epochs=5, batch_size=30)

img = cv2.imread(r"C:\Users\doant\Dropbox\dataset\0.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.reshape((1,100,100,1))
print(img.shape)
print(model.predict(x_train))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(acc, label = 'Accuracy')
plt.plot(val_acc, label = 'Validation Accuracy')

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss, label = 'loss')
plt.plot(val_loss, label = 'Validation loss')
plt.legend()

plt.show()