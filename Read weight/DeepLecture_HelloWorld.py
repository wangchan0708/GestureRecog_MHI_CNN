import numpy as np
import glob
import random as rnd
from scipy import misc
from PIL import Image

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
#from keras.utils import np_utils
#from keras.datasets import mnist

#categorical_crossentropy

#
f = open("labels.txt","r")
s = f.read()
f.close()
labels = s.split("\n")
	

def load_data(folder):
    print("reading file from "+ folder + "...")
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    f = open("labels.txt","r")
    s = f.read()
    f.close()
    label_name = s.split("\n")
	
	#load training data
    for k in range(1,4):
        for j in range(3,6):
            for i in range(0,len(label_name)):
                path=folder +str(k)+"/freq_"+str(j)+"/"+str(i+1)+"_[1-8].png"
                for image_path in glob.glob(path):
                    image = misc.imread(image_path, flatten=True)
                    image=image.reshape((100,100,1))
                    image=image.astype('float32')
                    image = image/255
                    x_train.append(image)
                    out=[0.0]*len(label_name)
                    out[i]=1.0
                    y_train.append(out)
        	
        	#load testing data
            for i in range(0,len(label_name)):
                path=folder +str(k)+"/freq_"+str(j)+"/"+str(i+1)+"_9.png"
                for image_path in glob.glob(path):
                    image = misc.imread(image_path, flatten=True)
                    image=image.reshape((100,100,1))
                    image=image.astype('float32')
                    image = image/255
                    x_test.append(image)
                    out=[0.0]*len(label_name)
                    out[i]=1.0
                    y_test.append(out)
                path=folder +str(k)+"/freq_"+str(j)+"/"+str(i+1)+"_[1-2][0-9].png"
                for image_path in glob.glob(path):
                    image = misc.imread(image_path, flatten=True)
                    image=image.reshape((100,100,1))
                    image=image.astype('float32')
                    image = image/255
                    x_test.append(image)
                    out=[0.0]*len(label_name)
                    out[i]=1.0
                    y_test.append(out)
    
    
    return (x_train,y_train),(x_test,y_test)

(x_train,y_train),(x_test,y_test)=load_data("Intelligence/MHI_")
#(x_train3,y_train3)=load_data("FinalImages")
#for elem in x_train3:
#	x_train.append(elem)
#for elem in y_train3:
#	y_train.append(elem)
	
print("training...")


#indexes = [i for i in range(len(x_test))]
#rand = rnd.sample(indexes,k=10)
#rand = [3*i for i in range(len(x_train)/3)]
#for i in rand:
#   x_test.append(x_train[i])
#   y_test.append(y_train[i])
#
#rand.sort(reverse=True)
#for i in rand:
#   x_train.pop(i)
#   y_train.pop(i)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

model2 = Sequential()
model2.add(Conv2D(15,(7,7),input_shape=(100,100,1)))
model2.add(MaxPooling2D((2,2)))
model2.add(Conv2D(30,(5,5)))
model2.add(MaxPooling2D((2,2)))
model2.add(Conv2D(50,(3,3)))
model2.add(MaxPooling2D((2,2)))
model2.add(Flatten())
model2.add(Dense(units=70,activation='relu'))
model2.add(Dense(units=14,activation='softmax'))
model2.summary()


model2.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model2.fit(x_train,y_train,batch_size=10,epochs=5)

score = model2.evaluate(x_train,y_train)
print ('\nTrain Acc:', score[1])
score = model2.evaluate(x_test,y_test)
print ('\nTest Acc:', score[1])

#y_pred = model2.predict(x_test)
#y_pred = np.argmax(y_pred,axis = 1)
#y_true = np.argmax(y_test,axis = 1)
#
#confusion_matrix = np.zeros((len(labels),len(labels)))
#for i in range(0,y_true.shape[0]):
#    confusion_matrix[y_true[i]][y_pred[i]]+=1;
#
#print('===============confusion_matrix==============')
#print confusion_matrix


#model2.save('my_model.h5')
#model2.save_weights('my_model_weights.h5')
#del model2

new_show = [0,3,7,19,21,23,25,27,29,31,33,35,37,39]

model = Sequential()
model.add(Conv2D(15,(7,7),input_shape=(100,100,1), weights=model2.layers[0].get_weights()))
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

for i in range(0,len(new_show)):
    c = np.asarray([x_test[new_show[i]]])
    activations = model.predict(c)
    activations = activations[0].reshape(94,94)
    img = Image.fromarray(activations, 'L')
    img.save('first_'+str(i)+'.png')


#model3 = Sequential()
#model3.add(Conv2D(15,(7,7),input_shape=(100,100,1), weights=model2.layers[0].get_weights()))
#model3.add(MaxPooling2D((2,2)))
#model3.add(Conv2D(30,(5,5), weights=model2.layers[1].get_weights()))
#model3.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
#
#model4 = Sequential()
#model4.add(Conv2D(15,(7,7),input_shape=(100,100,1), weights=model2.layers[0].get_weights()))
#model4.add(MaxPooling2D((2,2)))
#model4.add(Conv2D(30,(5,5), weights=model2.layers[1].get_weights()))
#model4.add(MaxPooling2D((2,2)))
#model4.add(Conv2D(50,(3,3), weights=model2.layers[2].get_weights()))
#model4.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
