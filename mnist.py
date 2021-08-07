import os
# import random
import numpy as np 
# import pandas as pd
# from numpy import genfromtxt
# import csv
import math
# from mpmath import mpf, mpc, mp
from matplotlib import pyplot as plt 
import matplotlib.image as mpimg 
# import pathlib
import imageio
# import numpy as np
import json
import argparse
from glob import glob
from PIL import Image
from numpy import asarray 

def predict_y(degree, alpha, iterations, theta0, x, y):
    m=y.size
    # print(m)
    j=0
    theta=np.matrix(theta0[1:])
    # print(theta)
    x1=x
    for j in range(degree):         #theta is a row vector
        
        if(j!=0):
            theta=np.append(theta, theta0[1:],axis=0)
            x=np.append(x, np.power(x1,j), axis=1)

    # theta=theta.T
    # print(theta)
    # theta=np.c_[1, theta]
    theta=np.append([1],theta)
    theta=np.matrix(theta)
    theta=theta.T
    # x = input_data[:,:6]
    # print("x= ",x.T)
    # print("x.t= ",x.T)
    mean=x.mean()
    std=x.std()
    x = (x - x.mean()) / x.std()
    x = np.c_[np.ones(x.shape[0]), x]
    # print(theta)
    iteration_array=np.array([])
    cost_array=np.array([])
    j=0
    for i in range(iterations):
        # print("theta.shape[0]=",theta.shape[0])
        # print("theta= ",theta)
        # print("theta= ",theta)
        prediction=1.0/(1 + np.exp(-np.dot(x,theta)))
        # prediction=prediction.T
        # print("prediction= ",prediction)
        # prediction=np.dot(x,theta.T)
        # if(j==0):
        #     print(prediction)
        error=prediction-y
        # if(i==0):
            # print("error.shape= ",error.shape[0]," x.t.shape= ",x.T.shape[1])
            # print(x.T)
        # cost=(1/(2*m))*np.dot(error.T,error)
        # cost_array=np.append(cost_array, np.array(cost))
        # iteration_array=np.append(iteration_array, np.array(i))
        # print("X.T= ",x.T)
        # print("error=",error)
        # if(j==1):
        #     print("theta= ",theta)
            # print()
            # print("x.T=",x.T)
            # print("error= ",error)
            # print("dot.shape= ",np.dot(x.T, error).shape[0],"dot= ",(alpha * (1/m) * np.dot(x.T, error)))
        theta=theta-(alpha * (1/m) * np.dot(x.T, error))
        j=j+1

    # print("cost_array.shape= ",cost_array.shape[0])
    # x1=np.matrix(test_data[:,:6])         #do transpose after it
    # x1=x1.astype('float64')
    # # print("y= ",y) 
    # x1 = (x1 - mean) / std
    # print(y1)
    # prediction=test(degree,x1,theta,mean,std)
    # y1=test_data[:,6]
    # print("y,shape= ",y.shape[0]," prediction.shape= ",prediction.shape[1])
    # print("iteration_array= ",iteration_array)
    # print("cost_array= ",cost_array)
    # plt.title("Matplotlib demo") 
    # plt.xlabel("No of iterations ") 
    # plt.ylabel("Loss function value") 
    # plt.plot(iteration_array,cost_array)            #0- circle, 1- upward parabola, 
    # x2=y
    # plt.plot(y,y,label="45deg")
    # plt.show() 
    return theta,mean,std

def test(degree,x, theta,mean, std):
    x1=x
    for j in range(degree):         #theta is a row vector
        
        if(j!=0):
            # theta=np.append(theta, theta0[1:],axis=0)
            x=np.append(x, np.power(x1,j), axis=1)

    x = (x - x.mean()) / x.std()
    x = np.c_[np.ones(x.shape[0]), x]

    prediction=1.0/(1 + np.exp(-np.dot(x,theta)))


    return prediction

if __name__ == "__main__":

# Import the necessary libraries 
# from PIL import Image 

  
  
# load the image and convert into 
# numpy array 
    # s='000000.jpeg'
    # img = Image.open(s) 
    # np_img = np.array(img)


    # # print(np_img[32,32])
    # print("before [",np_img.shape[0],"] x [",np_img.shape[1])
    # # np_img=np_img.reshape(1,np_img.shape[0]*np_img.shape[1])
    # np_img=np_img.flatten()
    # # print("now [",np_img.shape[0],"] x [",np_img.shape[1])
    # # print(np_img[0,32*64])
    # print(np_img[32*38])
    # degree=1
    # theta=np.random.random_sample(size=7)
    # print(theta)

    # x = input_data[:,:6]

    # y=np.ones(len(np_img)).T
    # theta,mean,std=predict_y(degree,0.1, 10000, theta, x,y)



    # img_dir = '/home/ashwini/Desktop/1st%20sem%202020-21/ELL409/Assignment1/Medical_MNIST/AbdomenCT/' # Enter Directory of all images  
    # data_path = os.path.join(img_dir,'*.jpeg') 
    # files = glob.glob(data_path)  
    # data = [] 
    # print("adf")
    # for f1 in files:
    #     print(f1)
    #     img = Image.open(f1) 
    #     np_img = np.array(img)

    # file_list=glob('Medical_MNIST/AbdomenCT/*.*')
    # print(file_list)

    # img_files = sorted(glob(os.path.join("Medical_MNIST/ChestCT/*.*"))) 
    img_files = sorted(glob(os.path.join("Chest/*.*"))) 
    img = Image.open(img_files[0]) 
    np_img = np.array(img)
    np_img=np.array([np_img.flatten()]) 
    x=np_img
    x1=np_img
    for fimg in img_files:
        print(fimg)
        if(fimg!=0):
            img = Image.open(fimg) 
            np_img = np.array(img)
            np_img=np.array([np_img.flatten()])
            # print(np_img)
            # print(fimg)
            x=np.append(x, np.array(np_img),axis=0)

    x=np.delete(x,0,0)
    y=np.array([np.ones(len(x))]).T
    # print(y)
    # print(np_img.shape[0])
    # print("x= ",x)

    theta=np.random.random_sample(size=(x.shape[1]+1))
    # print(theta.shape[0])
    degree=1
    theta,mean,std=predict_y(degree,0.1, 200, theta, x,y)

    print("theta= ",theta)

    img_files = sorted(glob(os.path.join("Hand/*.*"))) 
    img = Image.open(img_files[0]) 
    np_img = np.array(img)
    np_img=np.array([np_img.flatten()]) 
    # x=np_img
    x1=np_img
    for fimg in img_files:
        # print(fimg)
        if(fimg!=0):
            img = Image.open(fimg) 
            np_img = np.array(img)
            np_img=np.array([np_img.flatten()])
            # print(np_img)
            # print(fimg)
            x1=np.append(x1, np.array(np_img),axis=0)

    x1=np.delete(x1,0,0)
    y1=np.zeros(len(x)).T
    # # print("y= ",y) 
    # x1 = (x1 - mean) / std
    # print(y1)
    prediction=test(degree,x1,theta,mean,std)
    i=0
    for i in range(prediction.shape[0]):
        if(prediction[i,0]>=0.5):
            prediction[i,0]=1
        else:
            prediction[i,0]=0
    # print(prediction)
    i=0
    accuracy=0
    for i in range(y1.shape[0]):
        print("y1= ",y1[i]," prediction= ",prediction[i])
        if(y1[i]==prediction[i]):
            accuracy+=1

    accuracy=accuracy/y1.shape[0]
    print("accuracy= ",accuracy)
    # print("x.shape= ",x.shape[0])
    # print("adsfdfsdf")
    # for name in glob.glob('/home/ashwini/Desktop/1st%20sem%202020-21/ELL409/Assignment1/Medical_MNIST/AbdomenCT/*'): 
    #     print(name) 

    # parser = argparse.ArgumentParser(description='Assignment1_Medical_MNIST')
    # parser.add_argument('-i', '--img_path', type=str, default='img', required=True, help="Path for the image folder")
    # # parser.add_argument('-d', '--det_path', type=str, default='det', required=True, help="Path for the detected masks folder")
    # args = parser.parse_args()

    # # read the image files and arrange them in ascending order according to name.
    # img_files = sorted(glob(os.path.join(args.img_path, "*jpg")))