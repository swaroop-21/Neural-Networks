import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras import applications
from keras import optimizers
import cv2
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import rmsprop
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization











img_size_1=128
img_size_2=128
WHITE=[255,255,255]

images = []

for i in glob.glob("bed_jpg/*.jpg"):
    #img1=cv2.imread(i)
    #constant= cv2.copyMakeBorder(img1,108,108,0,0,cv2.BORDER_CONSTANT,value=WHITE)
    #plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
    images.append(cv2.imread(i))
images = np.asarray(images)
    
#images = np.asarray(images)
bed = []
for i in images :
    x = cv2.resize(i,(img_size_1,img_size_2))
    z = []
    z.append(x)
    z.append(0)
    bed.append(z)
bed = np.asarray(bed)
print(bed.shape)

images = []
for i in glob.glob("chair_jpg/*.jpg"):
    images.append(cv2.imread(i))
images = np.asarray(images)
chair = []
for i in images :
    x = cv2.resize(i,(img_size_1,img_size_2))
    z = []
    z.append(x)
    z.append(1)
    chair.append(z)

chair = np.asarray(chair)
print(chair.shape)

images = []
for i in glob.glob("lamp_jpg/*.jpg"):
    images.append(cv2.imread(i))
images = np.asarray(images)
lamp = []
for i in images :
    x = cv2.resize(i,(img_size_1,img_size_2))
    z = []
    z.append(x)
    z.append(2)
    lamp.append(z)
lamp = np.asarray(lamp)
print(lamp.shape)

images = []
for i in glob.glob("shelf_jpg/*.jpg"):
    images.append(cv2.imread(i))
images = np.asarray(images)
shelf = []
for i in images :
    x = cv2.resize(i,(img_size_1,img_size_2))
    z = []
    z.append(x)
    z.append(3)
    shelf.append(z)
shelf= np.asarray(shelf)
print(shelf.shape)

images = []
for i in glob.glob("sofa_jpg/*.jpg"):
    images.append(cv2.imread(i))
images = np.asarray(images)
sofa = []
for i in images :
    x = cv2.resize(i,(img_size_1,img_size_2))
    z = []
    z.append(x)
    z.append(4)
    sofa.append(z)
sofa = np.asarray(sofa)
print(sofa.shape)


images = []
for i in glob.glob("stool_jpg/*.jpg"):
    images.append(cv2.imread(i))
images = np.asarray(images)
stool = []
for i in images :
    x = cv2.resize(i,(img_size_1,img_size_2))
                     
    z = []
    z.append(x)
    z.append(5)
    stool.append(z)
stool = np.asarray(stool)
print(stool.shape)


images = []
for i in glob.glob("table_jpg/*.jpg"):
    images.append(cv2.imread(i))
images = np.asarray(images)
table = []
for i in images :
    x = cv2.resize(i,(img_size_1,img_size_2))
    z = []
    z.append(x)
    z.append(6)
    table.append(z)
table = np.asarray(table)
print(table.shape)


images = []
for i in glob.glob("wardrobe_jpg/*.jpg"):
    images.append(cv2.imread(i))
    
images = np.asarray(images)
wardrobe = []
for i in images :
    x = cv2.resize(i,(img_size_1,img_size_2))
    z = []
    z.append(x)
    z.append(7)
    wardrobe.append(z)
wardrobe = np.asarray(wardrobe)

print(wardrobe.shape)




final_list=[]
for i in chair:     
    final_list.append(i)
for i in bed:
    final_list.append(i)
for i in lamp:
    final_list.append(i)

for i in stool:
    final_list.append(i)

for i in sofa:
    final_list.append(i)
for i in shelf:
    final_list.append(i)
for i in wardrobe:
    final_list.append(i)
for i in table:
    final_list.append(i)    


final_list=np.asarray(final_list)
np.random.shuffle(final_list)





def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    one_hot = []
    a = np.array([0,0,0,0,0,0,0,0])
    a[x] = 1
    one_hot.append(a)
    return np.asarray(one_hot).astype('int32')







x=[]
y=[]
j=0
for i in final_list:
    x.append(final_list[j][0])
    y.append(one_hot_encode(final_list[j][1]))
    j+=1

x=np.asarray(x)
print(x.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



y_test=y_test.reshape((y_test.shape[0],y_test.shape[2]))
y_train=y_train.reshape((y_train.shape[0],y_train.shape[2]))
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
X_train=X_train/255
X_test=X_test/255

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)



model = Sequential()
model.add(Conv2D(32, (3, 3),input_shape=(img_size_2,img_size_1,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.25))
# model.add(Conv2D(256, (3, 3)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))

# model.add(Conv2D(256, (3, 5)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
# # model.add(Conv2D(256, (3, 3)))
# # model.add(BatchNormalization())
# # model.add(Activation('relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())


model.add(Dense(1000)) 
model.add(BatchNormalization())
model.add(Activation('relu'))  
# model.add(Dropout(0.3))    
# model.add(Dropout(0.5))   
# model.add(Dense(512))
# model.add(BatchNormalization())

# model.add(Activation('relu'))  
model.add(Dropout(0.5))    
# model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Activation('softmax'))

opt = keras.optimizers.Adam(lr=0.00005)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

batch_size=32
epochs=25




model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_test,y_test)  )

