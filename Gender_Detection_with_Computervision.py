#!/usr/bin/env python


from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import numpy as np
import cv2
import glob


#convert 3-dimensional image in to 2-dimensional array.
#convert one image in to structure form.

X=[]
img=cv2.imread("/home/abc/gender_training/train/female/female_0.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
X.append(gray.flatten())
X

#Use glob library to take all the images in female folder
y=[]
female_imgs=glob.glob("/home/abc/Documents/dataset/gender_training/train/female/*.*") 
for img in female_imgs: 
    img_3d=cv2.imread(img)
    gray_2d=cv2.cvtColor(img_3d,cv2.COLOR_BGR2GRAY)
    gray_2d=cv2.resize(gray_2d,(90,90))  
    X.append(gray_2d.flatten()) 
    y.append('female')

#add male images value with female image value.

male_imgs=glob.glob("/home/abc/Documents/dataset/gender_training/train/male/*.*")
for img in male_imgs:
    img_3d=cv2.imread(img)
    gray_2d=cv2.cvtColor(img_3d,cv2.COLOR_BGR2GRAY)
    gray_2d=cv2.resize(gray_2d,(90,90))
    X.append(gray_2d.flatten())
    y.append('male')

#using numpy convert it in to  two dimensional array because numpy has no append method only list has.
X_new=np.array(X)
X_new.shape

#contain value of all images
X_new

#convert all values in to 0 and 1
X_new=X_new/255
X_new.shape


#principal component analysis:technique convert large number of feature in to less number according to their importance and its value using eigenvalue and vector
pca=PCA(.99)
X_new_pca=pca.fit_transform(X_new)


#In new feature ,first feature conatin most of the information of dataset.
X_new_pca.shape

#apply logistic regression model
genderModel=LogisticRegression()
genderModel.fit(X_new_pca,y)

#the image that we want to predict first convert to gray and than only extract face and than it pass to model

#testing our model with some random image
img=cv2.imread("sampleimage.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faceModel=cv2.CascadeClassifier("/home/abc/Documents/dataset/trainedmodel/haarcascade_frontalface_default.xml")
faces=faceModel.detectMultiScale(gray)
for x,y,w,h in faces:
    face=gray[y:y+h,x:x+w]
    #cv2.imwrite("face.png",face)
    face=cv2.resize(face,(90,90))
    #print(face.shape)
    face=face.flatten()
    #print(face.shape)
    face=face/255
    face_pca=pca.transform([face])
    #print(face_pca.shape)
    print(genderModel.predict(face_pca),genderModel.predict_proba(face_pca))




