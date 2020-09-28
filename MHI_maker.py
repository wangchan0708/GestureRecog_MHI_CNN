word_n    = 0
folder  = "Data/Videos"
folder1 = "Data/MHIs"
sample_n =  1

import cv2
import numpy as np

begin = 0
end = 1000

f = open("labels.txt","r")
s = f.read()
f.close()
labels = s.split("\n")

while(True):
    word = labels[word_n]
    path = folder + '/' + str(word_n) + '/' + word + '_' + str(sample_n) +'.mov'
    cap = cv2.VideoCapture(path)
    ret,frame = cap.read()
    cap.release()
    if(ret==False and sample_n==1):
        print("Finished")
        break
    if(ret==False):
        word_n += 1
        sample_n = 1
        word
        continue
    print("Beginning word '" + word + "', sample n. " + str(sample_n))
    s = frame.shape

    h = s[0]/2
    w = s[1]-1

    xb=0
    yb=0

    MHI = np.zeros((h,w),  np.uint8)

    threshold = 200

    frame = []
    cap = cv2.VideoCapture(path)
    C = 0

    freq = 5

    while(cap.isOpened()):
        C+=1
        
        old_frame = frame
        ret, frame = cap.read()

        
        if(ret==False):
            break
        
        frame = cv2.medianBlur(frame,5)
        frame = cv2.GaussianBlur(frame,(5,5),0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.Canny(frame, 10, 130)
        #frame = cv2.medianBlur(frame,1)
        if(C<begin or C>end):
            old_frame = []
            continue
        #cv2.imshow('frame1', frame)
        #cv2.imshow('frame', MHI)
        #cv2.waitKey(2);
        if(C%freq!=0):
            continue
        if(old_frame==[]):
            continue
        for i in range(h):
            for j in range(w):
                if(abs(old_frame[i+xb,j+yb]-frame[i+xb,j+yb])>threshold):
                    MHI[i,j] = min(255, 255)
                else:
                    MHI[i,j] = max(int(MHI[i,j]-5), 0)

    MHI_path = path = folder1 + '/' + str(word_n) + '/' + str(word_n) + '_MHI_' + str(sample_n) + ".png"
    cv2.imwrite(MHI_path, MHI)
    print("Saved in: " + MHI_path)
    sample_n += 1
