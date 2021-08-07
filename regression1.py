import random
import numpy as np 
import pandas as pd
from numpy import genfromtxt
import csv
import math
from mpmath import mpf, mpc, mp
from matplotlib import pyplot as plt 

def h(theta,x):
	z=(theta.transpose())@x
	# print(z)
	return z

if __name__ == "__main__":

		df=pd.read_excel("weather_data.xlsx")
		input_data=np.array(df.values.tolist())		
		# print(np.amin(input_data[:,0]))

		#plotting 0th feature with output
		# x1=input_data[:,2]
		y=input_data[:,6]

		# for row in range(len(input_data)):
		# 	print("x1[",row,"]= ",x1[row]," y1[",row,"]= ",y1[row])
		# print("max t= ",np.amax(y1)," max de= ",np.amax(x1))
		# plt.title("Matplotlib demo") 
		# plt.xlabel("x axis caption") 
		# plt.ylabel("y axis caption") 
		# plt.scatter(x1,y1) 			#0- circle, 1- upward parabola, 
		# plt.show() 
		
		theta=np.random.random_sample(size = 6)
		x = input_data[:,:5]
		# print("x= ",x)
		
		# print(x.mean())
		x = (x - x.mean()) / x.std()
		# print("mean1= ",np.mean(input_data[:,0])," x= ",x)
		x = np.c_[np.ones(x.shape[0]), x] 
		# print('now x is ',x)
		alpha = 0.01  
		iterations = 2000 
		m = y.size 
		cost = 0

		for i in range(iterations):
			prediction=np.dot(x,theta)
			error=prediction-y
			cost=(1/(2*m))*np.dot(error.T,error)
			theta=theta-(alpha * (1/m) * np.dot(x.T, error))
	    	# prediction = np.dot(x, theta)
	    	# error = prediction - y
	    	# cost = 1/(2*m) * np.dot(error.T, error)
	    	# theta = theta - (alpha * (1/m) * np.dot(x.T, error))

		print("theta= ",theta)
		# for row in range(len(input_data)):
		# 	print(input_data[row])
		y=np.matrix(input_data[0][0:len(input_data[0])-1]).transpose()			#do transpose after it
		y=y.astype('float64') 
		y = (y - y.mean()) / y.std()
		y = np.c_[np.ones(y.shape[0]), y]
		print(theta.transpose()@y)			
		# print((input_data[0,0:(len(input_data[0])-1)]))
		