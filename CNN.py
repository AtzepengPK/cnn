from keras.models import Sequential

from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import numpy as np

classifier = Sequential()

classifier.add(Convolution2D(
    32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(
    32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'C:/Users/sergiu.axinte/Desktop/python/cnn/dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'C:/Users/sergiu.axinte/Desktop/python/cnn/dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

metrics = classifier.fit_generator(
    training_set,
    steps_per_epoch=8000,
    epochs=2,
    validation_data=test_set,
    validation_steps=2000)

img1 = load_img(
    path=r"C:\Users\sergiu.axinte\Desktop\python\cnn\dataset\single_prediction\cat_or_dog_1.jpg", target_size=(64, 64))
img2 = load_img(
    path=r"C:\Users\sergiu.axinte\Desktop\python\cnn\dataset\single_prediction\cat_or_dog_2.jpg", target_size=(64, 64))
img3 = load_img(
    path=r"C:\Users\sergiu.axinte\Pictures\images.jfif", target_size=(64, 64))
img1 = img_to_array(img1)
img2 = img_to_array(img2)
img3 = img_to_array(img3)

predict_set = np.array([img1, img2, img3])
result = classifier.predict_classes(predict_set)

for idx, image in enumerate(result):
    print('img%d : %s' % (idx+1, 'dog' if image == 1 else 'cat'))

classifier.eval
