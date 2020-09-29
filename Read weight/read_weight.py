import numpy as np
import h5py
from PIL import Image

f = h5py.File('my_model.h5', 'r')


# read first layer weighting
my_array = f['model_weights']['conv2d_3']['conv2d_3']['kernel'][()]

filiter_size=3
filiter_num=50

first_filiter=np.zeros((filiter_num,filiter_size,filiter_size), dtype=np.uint8)

for i in range(0,filiter_num):
    filiter=np.zeros((filiter_size,filiter_size))
    for j in range(0,filiter_size):
        for k in range(0,filiter_size):
            filiter[j][k]=my_array[j][k][0][i]
    max_array=np.amax(filiter)
    min_array=np.amin(filiter)
    for j in range(0,filiter_size):
        for k in range(0,filiter_size):
            first_filiter[i][j][k]=( filiter[j][k]-min_array ) / (max_array - min_array) * 255
    img = Image.fromarray(first_filiter[i], 'L')
    img.save('third_'+str(i)+'.png')