import os
import cv2
from keras.models import Sequential
from keras.layers import Conv2D,Dropout,MaxPooling2D,Flatten,Dense,BatchNormalization
from keras import initializers
from keras.optimizers import Adam
import numpy as np
from keras.utils import np_utils

label = {"rock": 0,"paper": 1,"scissors": 2}
img_path = 'dataset/train'
df = []
def ret_func(val):
    return label[val]

for dirs in os.listdir(img_path):
    path = os.path.join(img_path, dirs)
    for i in os.listdir(path):
        img = cv2.imread(os.path.join(path, i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        df.append([img, dirs])
img_data, img_labels = zip(*df)
img_labels = list(map(ret_func, img_labels))
img_labels = np_utils.to_categorical(img_labels)


input_size = (256,256,3)    
acti = 'relu'
opt= Adam(learning_rate=0.001)

def create_func():
    model = Sequential()

    model.add(Conv2D(32,(3,3),padding='same' ,kernel_initializer=initializers.VarianceScaling(),
                     input_shape=input_size,activation=acti,strides=(1, 1)))
    
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
   
    model.add(Conv2D(64,(3,3),padding='same',kernel_initializer=initializers.VarianceScaling(),
                     activation=acti,strides=(1, 1)))
    
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
   
    model.add(Conv2D(128,(3,3),padding='same',kernel_initializer=initializers.VarianceScaling(),
                     activation=acti,strides=(1, 1)))
    
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    

    model.add(Flatten())

    model.add(Dense(3,activation='softmax'))
    model.compile(optimizer = opt , loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

model = create_func()
model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(np.array(img_data), np.array(img_labels), epochs=55,batch_size=10)
model.save("finalmodel.h5")