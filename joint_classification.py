import os
from keras.preprocessing.text import one_hot, Tokenizer
import numpy as np
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout
from random import shuffle
import cv2
from keras import applications
from keras.layers import Dense, GlobalAveragePooling2D, merge, Merge, Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from PIL import Image
import random

x_train = []
y_train = []

x_train_shuf = []
y_train_shuf = []

x_train_1 = []
x_train_2 = []

max_words = 10000

epochs = 50
batch_size = 32

training_dir = "/home/aminnaar/viglink_images/combined"
sub_dirs = [dir for dir in os.listdir(training_dir) if ".txt" not in dir and ".DS_Store" not in dir]

class_counter = 0
for dir in sub_dirs:
    full_path = training_dir + "/" + dir
    text_files = [f for f in os.listdir(full_path) if ".txt" in f]
    # print text_files
    print dir
    for tf in text_files:
        r = random.uniform(0, 1)
        if r < 0.3:
            continue
        file_root = tf.split(".")[0]
        image_filename = full_path + "/" + file_root + ".jpg"
        # print image_filename
        image = cv2.resize(cv2.imread(image_filename), (299, 299)).astype(
            np.float32)  # cv2.imread(full_path + "/" + file_root + ".jpg")
        # resized = cv2.resize(image,(299,299))
        # x_train_1.append(image)
        contents = open(full_path + "/" + tf, "r").read()

        x_train.append((one_hot(text=contents, n=max_words, lower=True, split=" "), image))
        y_train.append(class_counter)

    class_counter += 1

num_classes = np.max(y_train) + 1

index_shuf = range(len(y_train))
shuffle(index_shuf)
for i in index_shuf:
    x_train_shuf.append(x_train[i])
    y_train_shuf.append(y_train[i])

x_train_text = [w[0] for w in x_train_shuf]
x_train_image = [z[1] for z in x_train_shuf]
# test_image_arr=np.array(x_train_image)
#
# blah = np.array(x_train_1)
#
# print "image input array info"
# print blah.shape
# print blah[0],type(blah[0])
# print blah[0].shape

print len(x_train)
print len(y_train)
print(num_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words)
x_train_shuf = tokenizer.sequences_to_matrix(x_train_text, mode='binary')
print('x_train shape:', x_train_shuf.shape)

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train_shuf = to_categorical(y_train_shuf, num_classes)
print('y_train shape:', y_train_shuf.shape)

print('Building model...')
#
# branch_1 = Sequential()
# branch_1.add(Dense(512, input_shape=(max_words,), activation='relu'))

text_inputs = Input(shape=(max_words,))
branch_1 = Dense(512, activation='relu')(text_inputs)

# create the base pre-trained model
base_model = applications.InceptionV3(weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False

# add a global spatial average pooling layer
branch_2 = base_model.output
branch_2 = GlobalAveragePooling2D()(branch_2)
# let's add a fully-connected layer
# branch_2 = Dense(256, activation='relu')(branch_2)
branch_2 = Dropout(0.5)(branch_2)
branch_2 = Dense(256, activation='sigmoid')(branch_2)

joint = merge([branch_1, branch_2], mode='concat')
# joint = Dense(512, activation='relu')(joint)
joint = Dropout(0.5)(joint)
predictions = Dense(num_classes, activation='softmax')(joint)

full_model = Model(inputs=[base_model.input, text_inputs], outputs=[predictions])

full_model.compile(loss='categorical_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])

# print np.array(x_train_image).shape
# print np.array(x_train_shuf).shape
#
# print np.array(x_train_image)[0].shape
# print np.array(x_train_shuf)[0].shape
#
# print type(x_train_image)
x_images = np.array(x_train_image)
x_text = np.array(x_train_shuf)
y = np.array(y_train_shuf)

for layer in full_model.layers:
    print layer

history = full_model.fit([x_images, x_text], y,
                         epochs=epochs, batch_size=batch_size,
                         verbose=1, validation_split=0.2, shuffle=True)

full_model.save('model2.h5')

