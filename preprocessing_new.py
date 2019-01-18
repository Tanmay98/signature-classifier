import cv2
import numpy as np
import os
import pandas as pd
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from skimage import img_as_float
from skimage import io, color, morphology
from skimage import img_as_float
from scipy.stats import kurtosis,skew
import pickle

def spatial(im):
    image1 = img_as_float(color.rgb2gray(io.imread(im)))
    #image1=img_as_float(gray_image3)
    image_binary = image1 < 0.5
    out_skeletonize = morphology.skeletonize(image_binary)
    out_thin = morphology.thin(image_binary)

    #out_thin.shape[:-1] is row
    #out_thin.shape[1:] is column

    column_size = out_thin.shape[1]
    row_size = out_thin.shape[0]

    #change the true to 1 and false to 0 in the array

    spatial_symbols = []
    for row in range(int(row_size)):
            if (row == 0):
                    for column in range(1, int(column_size)-1):
                            if (out_thin[row,column] == 1):
                                    if (out_thin[row,column-1] == 1 and out_thin[row,column+1] == 1 and out_thin[row+1][column] == 1):
                                            spatial_symbols.append(out_thin[row,column])

            elif row == int(row_size):
                    for column in range(1, int(column_size)-1):
                            if (out_thin[row,column] == 1):
                                    if (out_thin[row,column-1] == 1 and out_thin[row,column+1] == 1 and out_thin[row-1][column] == 1):
                                            spatial_symbols.append(out_thin[column])
            else:
                    for column in range(1,int(column_size)-1):
                            if out_thin[row,column] == 1:
                                    if (out_thin[row,column-1] == 1 and out_thin[row,column+1] == 1 and out_thin[row+1][column] == 1 and out_thin[row-1][column] == 1):
                                            spatial_symbols.append(out_thin[row,column])
                                    elif (out_thin[row,column-1] == 0 and out_thin[row,column+1] == 1 and out_thin[row+1][column] == 1 and out_thin[row-1][column] == 1):
                                            spatial_symbols.append(out_thin[row,column])
                                    elif (out_thin[row,column-1] == 1 and out_thin[row,column+1] == 0 and out_thin[row+1][column] == 1 and out_thin[row-1][column] == 1):
                                            spatial_symbols.append(out_thin[row,column])
                                    elif (out_thin[row,column-1] == 1 and out_thin[row,column+1] == 1 and out_thin[row+1][column] == 0 and out_thin[row-1][column] == 1):
                                            spatial_symbols.append(out_thin[row,column])
                                    elif (out_thin[row,column-1] == 1 and out_thin[row,column+1] == 1 and out_thin[row+1][column] == 1 and out_thin[row-1][column] == 0):
                                            spatial_symbols.append(out_thin[row,column])
                            else:
                                    pass
    return spatial_symbols

ls_train=[]
#0=forge 1=genuine

#dutch genuine
s=np.zeros(shape=[239,6],dtype="float32")
count=0
image_folder = "E:/DL/signature_classifier/trainingSet/OfflineSignatures/Dutch/TrainingSet/OfflineGenuine"
files=os.listdir(image_folder)
files=list(map(lambda x: os.path.join(image_folder,x),files))
flag = 0
labels = []

a=(len(files))
for i in range(a):
    im = (files[i])
    l_string=int(im[-10:-7])
    labels.append(l_string)

    img=cv2.imread(im)
    img = cv2.resize(img, (50,100)) 
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm_image=cv2.normalize(gray_image,gray_image, 0, 255, cv2.NORM_MINMAX)
    ret, thresh = cv2.threshold(norm_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    sig_pixels=0
    for i in thresh:
        for j in i:
            if j==255:
                sig_pixels+=1
    density=(sig_pixels/(thresh.shape[0]*thresh.shape[1]))
    smooth_image = cv2.GaussianBlur(norm_image,(5,5),0)
    ku=kurtosis(smooth_image,axis=0)
    sk=skew(smooth_image,axis=0)
    sd=np.std(smooth_image,axis=0)
    spat=spatial(im)

## roi
    kernel = np.ones((5,5), np.uint8) 
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
#sort contours 
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs): 
        # Get bounding box 
        x, y, w, h = cv2.boundingRect(ctr) 
        
        # Getting ROI 
        roi = img[y:y+h, x:x+w] 
        # show ROI 
        #cv2.imshow('segment no:'+str(i),roi) 
        cv2.rectangle(img,(x,y),( x + w, y + h ),(0,255,0),2)
    ratio=(h/w)
    
    #cv2.waitKey(0) 
    s[count][0]=density    
    s[count][1]=len(spat)
    s[count][2]=np.mean(ku)
    s[count][3]=np.mean(sk)
    s[count][4]=np.mean(sd)
    s[count][5]=np.mean(ratio)
    
    count+=1
    flag+=1
    ls_train.append(smooth_image)
    print(flag)

s = pd.DataFrame(s)
s.columns = ["density","spatial symbols", "kurtoisis", "skewness", "stddev", "ratio"]
s["class labels"] = pd.Series(np.array(labels), index=s.index)
s["label"] = pd.Series(np.ones(239))

#dutch forgeries
t=np.zeros(shape=[123,6],dtype="float32")
count=0
image_folder = "E:/DL/signature_classifier/trainingSet/OfflineSignatures/Dutch/TrainingSet/OfflineForgeries"
files=os.listdir(image_folder)
files=list(map(lambda x: os.path.join(image_folder,x),files))
flag = 0
labels = []

a=(len(files))
for i in range(a):
    im = (files[i])
    l_string=int(im[-9:-7])
    labels.append(l_string)

    img=cv2.imread(im)
    img = cv2.resize(img, (50,100)) 
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm_image=cv2.normalize(gray_image,gray_image, 0, 255, cv2.NORM_MINMAX)
    ret, thresh = cv2.threshold(norm_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    sig_pixels=0
    for i in thresh:
        for j in i:
            if j==255:
                sig_pixels+=1
    density=(sig_pixels/(thresh.shape[0]*thresh.shape[1]))
    smooth_image = cv2.GaussianBlur(norm_image,(5,5),0)
    ku=kurtosis(smooth_image,axis=0)
    sk=skew(smooth_image,axis=0)
    sd=np.std(smooth_image,axis=0)
    spat=spatial(im)

## roi
    kernel = np.ones((5,5), np.uint8) 
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
#sort contours 
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs): 
        # Get bounding box 
        x, y, w, h = cv2.boundingRect(ctr) 
        
        # Getting ROI 
        roi = img[y:y+h, x:x+w] 
        # show ROI 
        #cv2.imshow('segment no:'+str(i),roi) 
        cv2.rectangle(img,(x,y),( x + w, y + h ),(0,255,0),2)
    ratio=(h/w)
    
    #cv2.waitKey(0) 
    t[count][0]=density    
    t[count][1]=len(spat)
    t[count][2]=np.mean(ku)
    t[count][3]=np.mean(sk)
    t[count][4]=np.mean(sd)
    t[count][5]=np.mean(ratio)
    
    count+=1
    flag+=1
    ls_train.append(smooth_image)
    print(flag)

t = pd.DataFrame(t)
t.columns = ["density","spatial symbols", "kurtoisis", "skewness", "stddev", "ratio"]
t["class labels"] = pd.Series(np.array(labels), index=t.index)
t["label"] = pd.Series(np.zeros(123))

#genuines
u=np.zeros(shape=[938,6],dtype="float32")
count=0
image_folder = "E:/DL/signature_classifier/genuines"
files=os.listdir(image_folder)
files=list(map(lambda x: os.path.join(image_folder,x),files))
flag = 0
labels = []

a=(len(files))
for i in range(a):
    im = (files[i])
    l_string = 16 + int(im[-7:-4])
    labels.append(l_string)

    img=cv2.imread(im)
    img = cv2.resize(img, (50,100)) 
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm_image=cv2.normalize(gray_image,gray_image, 0, 255, cv2.NORM_MINMAX)
    ret, thresh = cv2.threshold(norm_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    sig_pixels=0
    for i in thresh:
        for j in i:
            if j==255:
                sig_pixels+=1
    density=(sig_pixels/(thresh.shape[0]*thresh.shape[1]))
    smooth_image = cv2.GaussianBlur(norm_image,(5,5),0)
    ku=kurtosis(smooth_image,axis=0)
    sk=skew(smooth_image,axis=0)
    sd=np.std(smooth_image,axis=0)
    spat=spatial(im)

## roi
    kernel = np.ones((5,5), np.uint8) 
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
#sort contours 
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs): 
        # Get bounding box 
        x, y, w, h = cv2.boundingRect(ctr) 
        
        # Getting ROI 
        roi = img[y:y+h, x:x+w] 
        # show ROI 
        #cv2.imshow('segment no:'+str(i),roi) 
        cv2.rectangle(img,(x,y),( x + w, y + h ),(0,255,0),2)
    ratio=(h/w)
    
    #cv2.waitKey(0) 
    u[count][0]=density    
    u[count][1]=len(spat)
    u[count][2]=np.mean(ku)
    u[count][3]=np.mean(sk)
    u[count][4]=np.mean(sd)
    u[count][5]=np.mean(ratio)
    
    count+=1
    flag+=1
    ls_train.append(smooth_image)
    print(flag)

u = pd.DataFrame(u)
u.columns = ["density","spatial symbols", "kurtoisis", "skewness", "stddev", "ratio"]
u["class labels"] = pd.Series(np.array(labels), index=u.index)
u["label"] = pd.Series(np.ones(938))

#forgeries
v=np.zeros(shape=[624,6],dtype="float32")
count=0
image_folder = "E:/DL/signature_classifier/forgeries"
files=os.listdir(image_folder)
files=list(map(lambda x: os.path.join(image_folder,x),files))
flag = 0
labels = []

a=(len(files))
for i in range(a):
    im = (files[i])
    l_string = 16 + int(im[-6:-4])
    labels.append(l_string)

    img=cv2.imread(im)
    img = cv2.resize(img, (50,100)) 
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm_image=cv2.normalize(gray_image,gray_image, 0, 255, cv2.NORM_MINMAX)
    ret, thresh = cv2.threshold(norm_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    sig_pixels=0
    for i in thresh:
        for j in i:
            if j==255:
                sig_pixels+=1
    density=(sig_pixels/(thresh.shape[0]*thresh.shape[1]))
    smooth_image = cv2.GaussianBlur(norm_image,(5,5),0)
    ku=kurtosis(smooth_image,axis=0)
    sk=skew(smooth_image,axis=0)
    sd=np.std(smooth_image,axis=0)
    spat=spatial(im)

## roi
    kernel = np.ones((5,5), np.uint8) 
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
#sort contours 
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs): 
        # Get bounding box 
        x, y, w, h = cv2.boundingRect(ctr) 
        
        # Getting ROI 
        roi = img[y:y+h, x:x+w] 
        # show ROI 
        #cv2.imshow('segment no:'+str(i),roi) 
        cv2.rectangle(img,(x,y),( x + w, y + h ),(0,255,0),2)
    ratio=(h/w)
    
    #cv2.waitKey(0) 
    v[count][0]=density    
    v[count][1]=len(spat)
    v[count][2]=np.mean(ku)
    v[count][3]=np.mean(sk)
    v[count][4]=np.mean(sd)
    v[count][5]=np.mean(ratio)
    
    count+=1
    flag+=1
    ls_train.append(smooth_image)
    print(flag)

v = pd.DataFrame(v)
v.columns = ["density","spatial symbols", "kurtoisis", "skewness", "stddev", "ratio"]
v["class labels"] = pd.Series(np.array(labels), index=v.index)
v["label"] = pd.Series(np.zeros(624))

frames = [s,t]
frames2 = [u,v]
dataset = pd.concat(frames)
dataset2 = pd.concat(frames2)

frames3 = [dataset, dataset2]
full_dataset = pd.concat(frames3)

print(full_dataset.head())

pickle_out = open("finaldataset", "wb")
pickle.dump(full_dataset, pickle_out)
pickle_out.close()
 