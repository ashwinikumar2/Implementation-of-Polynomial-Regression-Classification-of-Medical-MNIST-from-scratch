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

# def cross_entropy(actual_output, predicted_output):
#     # print("actual_output= ",actual_output," predicted_output= ",predicted_output)
#     return (-(actual_output*math.log(predicted_output)+(1-actual_output)*math.log(1- predicted_output)))

def prediction_by_logistic_regression(degree, alpha, iterations, theta0, x, y):
    m=y.size
    j=0
    theta=np.matrix(theta0[1:])
    # print(theta)
    x1=x
    for j in range(degree):         #theta is a row vector
        
        if(j!=0):
            theta=np.append(theta, np.matrix(theta0[1:]),axis=0)
            x=np.append(x, np.power(x1,j), axis=1)
    theta=np.append([1],theta)
    theta=np.matrix(theta)
    theta=theta.T
    # print(theta)
    mean=x.mean()
    std=x.std()
    x = (x - x.mean()) / x.std()
    x = np.c_[np.ones(x.shape[0]), x]
    # print(theta)
    iteration_array=np.array([])
    cost_array=np.array([])
    j=0
    for i in range(iterations):
        prediction=1.0/(1 + np.exp(-np.dot(x,theta)))
        error=prediction-y
        _y = y.reshape(-1, 1)
        cost =(np.dot((-_y.T),np.log(prediction)) - np.dot(((1-_y).T),np.log(1-prediction)))
        cost_array=np.append(cost_array, np.array(cost))
        iteration_array=np.append(iteration_array, np.array(i))
        theta=theta-(alpha * (1/m) * np.dot(x.T, error))
        j=j+1

    # print("cost_array.shape= ",cost_array.shape[0]," iterations.shape= ",iteration_array.shape[0])
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

def test_for_logistic_regression(degree,x, theta,mean, std):
    x1=x
    for j in range(degree):         #theta is a row vector
        
        if(j!=0):
            # theta=np.append(theta, theta0[1:],axis=0)
            x=np.append(x, np.power(x1,j), axis=1)

    # print("x= ",x)
    x = (x - mean) / std
    x = np.c_[np.ones(x.shape[0]), x]

    prediction=1.0/(1 + np.exp(-np.dot(x,theta)))

    tpr_array=np.zeros(10)
    fpr_array=np.zeros(10)
    threshold=0.1
    t=0
    max_accuracy=0
    for t in range(10):             #Threshold variation for plot of fpr and tpr
        i=0
        for i in range(prediction.shape[0]):
            if(prediction[i,0]>=threshold):
                prediction[i,0]=1
            else:
                prediction[i,0]=0
        i=0
        accuracy=0.0
        false_negative=0.00
        false_positive=0.0
        true_positive=0.0
        true_negative=0.0
        for i in range(y1.shape[0]):
            if(y1[i]==prediction[i]):
                accuracy+=1

            if(y1[i]==1):
                true_positive+=1
            if(y1[i]==0):
                true_negative+=1
            if(prediction[i]==1 and y1[i]==0):
                false_positive+=1
            if(prediction[i]==0 and y1[i]==1):
                false_negative+=1

        accuracy=(float)(accuracy/y1.shape[0])
        max_accuracy=max(max_accuracy, accuracy)
        threshold+=0.1

    return prediction,max_accuracy

def prediction_by_naive_bayes(no_of_classes,x,y):

    #CALCULATING P(y) for each class
    # frequencies_of_each_class=np.zeros(no_of_classes)
    # print("frequencies_of_each_class= ",frequencies_of_each_class)
    # for output in range(y.shape[0]):
    #     class_of_y=(int)(y[output,0])
    #     frequencies_of_each_class[class_of_y]+=1

    # total_frequency=frequencies_of_each_class.sum()
    # frequencies_of_each_class=frequencies_of_each_class/total_frequency
    classes, frequency_of_each_class = np.unique(input_data[:,3],return_counts = True)
    print("frequency= ",frequency_of_each_class)

    #CALCULATING P(xi|y)

    for i in range(x.shape[1]):
        unique_feature_values, frequency_of_feature_value=np.unique(input_data[:,i],return_counts = True)

if __name__ == "__main__":
    input_data1 = np.genfromtxt("health_data(copy).csv", delimiter=",", skip_header=1)
    # training_size=(int)(0.7*len(input_data1))
    # print(training_size)
    print("input_data1= ",input_data1.shape[0])
    degree=1
    theta=np.random.random_sample(size=4)
    k=0
    k_fold=10
    net_accuracy_by_logistic_regression=0
    for k in range(k_fold):             #CROSS VALIDATION
        input_data= np.array([[]])

        inp=input_data1[(k+1)*(len(input_data1)//k_fold):(len(input_data1)),:]
        if(k!=0):
            input_data=input_data1[0:k*(len(input_data1)//k_fold),:]
            input_data=np.append(input_data, inp,axis=0)
        else:
            input_data=inp
        
        test_data=input_data1[k*(len(input_data1)//k_fold):(k+1)*(len(input_data1)//k_fold),:]
        #INPUT DATA 
        x = input_data[:,:3]
        y=np.matrix((input_data[:,3])).T

        #TEST DATA
        x1=test_data[:,:3]           
        y1=test_data[:,3]

        #LOGISTIC REGRESSION STARTS
        theta,mean,std=prediction_by_logistic_regression(degree,1, 1000, theta, x,y)
        prediction,max_accuracy=test_for_logistic_regression(degree,x1,theta,mean,std)
        net_accuracy_by_logistic_regression+=max_accuracy
        #LOGISTIC REGRESSION END

        #NAIVE BAYES 
        prediction_by_naive_bayes(2,x,y)

    net_accuracy_by_logistic_regression=net_accuracy_by_logistic_regression/k_fold
    # print("net_accuracy= ",net_accuracy)

    # print("2**2= ",2**2)
















# def h(t0, t1,x1, t2,x2, t3,x3):
#   theta=[[t0],[t1],[t2],[t3]]
#   theta1=np.array(theta)
#   x=np.array([[1],[x1],[x2],[x3]])
#   # print(theta1)
#   # print(theta1.transpose())
#   # print(x)
#   z= (np.dot((theta1.transpose()),x))[0][0]

#   val=np.longfloat(1/np.longfloat(1+np.longfloat(math.exp(-z))))
#   # print(mpf(1)+mpf(math.exp(-z)))
#   # print("val= ",val," z= ",z," math.exp=  ",math.exp(-z))
#   # print(np.longfloat(math.exp(-z)))
#   return val

# def cross_entropy(actual_output, predicted_output):
#   # print("actual_output= ",actual_output," predicted_output= ",predicted_output)
#   return (-(actual_output*math.log(predicted_output)+(1-actual_output)*math.log(1- predicted_output)))

# if __name__ == "__main__":

#       input_data = np.genfromtxt("health_data.csv", delimiter=",", skip_header=1)
                            
#       # print(len(my_data))                               #to get number of rows of input_data
#       # print(input_data[:,0])                                #to access a coloumn of a np array
#       # print(len(input_data))
#       # print(float(h(0,1,1,1,1,1,1)))
#       # mpf(200) + mpf(2e-26) + mpc(1j)
#       # x=2e-150
#       # print(x)
#       # print(np.longfloat(x))
#       # print(numpy.longfloat(2)+numpy.longfloat(2e-150))
#       # print(round(6.66666666666, 4))

#       print(np.sum(input_data[:,0]))

#       unique, frequency = np.unique(input_data[:,3],return_counts = True) 
#       total_frequency=frequency.sum()
#       # print(np.amin(input_data[:,0]))
#       #normalization
#       mean_x1=np.sum(input_data[:,0])/len(input_data)
#       mean_x2=np.sum(input_data[:,1])/len(input_data)
#       mean_x3=np.sum(input_data[:,2])/len(input_data)
#       max1=np.amax(input_data[:,0])
#       max2=np.amax(input_data[:,1])
#       max3=np.amax(input_data[:,2])
#       min1=np.amin(input_data[:,0])
#       min2=np.amin(input_data[:,1])
#       min3=np.amin(input_data[:,2])
#       loss=0

#       #theta construction
#       theta=np.dot((np.dot(np.linalg.inv(np.dot((input_data[:][0:2]).transpose(),(input_data[:][0:2])))),(input_data[:][0:2]).transpose()),)
#       # print("mean_x3= ",mean_x3," max2= ",max2," min1= ",min1)
        
#       #COST FUNCTION
#       for row in range(len(input_data)):
            
#           # print(input_data[row])
#           x1=(input_data[row][0]-mean_x1)/(max1-min1)
#           x2=(input_data[row][1]-mean_x2)/(max2-min2)
#           x3=(input_data[row][2]-mean_x3)/(max3-min3)
#           # print("x1= ",x1," x2= ",x2, "x3= ",x3)
#           function_value=((h(0,1,x1,1,x2,1,x3)))
#           label=0
#           if(function_value>=0.5):
#               label=1
#           else:
#               label=0

#           index=np.where(unique== input_data[row][3])
#           freq=frequency[index]

#           loss+=float(cross_entropy(freq/total_frequency, function_value))
#           # print(cross_entropy(freq/total_frequency, function_value))

#       print(loss)




        