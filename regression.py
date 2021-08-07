import os
import random
import numpy as np 
import pandas as pd
from numpy import genfromtxt
import csv
import math
from mpmath import mpf, mpc, mp
from matplotlib import pyplot as plt 

def predict_y(degree, alpha, iterations, theta0, x, y):
	m=y.size
	j=0
	theta=theta0[1:]
	x1=x
	for j in range(degree):			#theta is a row vector
		
		if(j!=0):
			theta=np.append(theta, theta0[1:],axis=0)
			x=np.append(x, np.power(x1,j), axis=1)

	theta=theta.transpose()
	# print(theta)
	# theta=np.c_[1, theta]
	theta=np.append([1],theta)
	# x = input_data[:,:6]
	# print("x= ",x.T)
	# print("x.t= ",x.T)
	mean=x.mean()
	std=x.std()
	x = (x - x.mean()) / x.std()
	x = np.c_[np.ones(x.shape[0]), x]
	iteration_array=np.array([])
	cost_array=np.array([])
	for i in range(iterations):
		prediction=np.dot(x,theta.T)
		error=prediction-y
		# if(i==0):
			# print("error.shape= ",error.shape[0]," x.t.shape= ",x.T.shape[1])
			# print(x.T)
		cost=(1/(2*m))*np.dot(error.T,error)
		cost_array=np.append(cost_array, np.array(cost))
		iteration_array=np.append(iteration_array, np.array(i))
		theta=theta-(alpha * (1/m) * np.dot(x.T, error))

	print("cost_array.shape= ",cost_array.shape[0])
	# x1=np.matrix(test_data[:,:6])			#do transpose after it
	# x1=x1.astype('float64')
	# # print("y= ",y) 
	# x1 = (x1 - mean) / std
	# print(y1)
	# prediction=test(degree,x1,theta,mean,std)
	# y1=test_data[:,6]
	# print("y,shape= ",y.shape[0]," prediction.shape= ",prediction.shape[1])
	print("iteration_array= ",iteration_array)
	print("cost_array= ",cost_array)
	plt.title("Matplotlib demo") 
	plt.xlabel("No of iterations ") 
	plt.ylabel("Loss function value") 
	plt.plot(iteration_array,cost_array) 			#0- circle, 1- upward parabola, 
	# x2=y
	# plt.plot(y,y,label="45deg")
	plt.show() 

	return theta,mean,std

def test(degree,x, theta,mean, std):
	x1=x
	for j in range(degree):			#theta is a row vector
		
		if(j!=0):
			# theta=np.append(theta, theta0[1:],axis=0)
			x=np.append(x, np.power(x1,j), axis=1)

	x = (x - x.mean()) / x.std()
	x = np.c_[np.ones(x.shape[0]), x]

	prediction=np.dot(x,theta.T)


	return prediction
	
if __name__ == "__main__":

		df=pd.read_excel("weather_data.xlsx")
		input_data1=np.array(df.values.tolist())		
		# print(np.amin(input_data[:,0]))
		# print((0.7*len(input_data1)))
		training_size=(int)(0.7*len(input_data1))
		input_data=input_data1[:training_size,:]
		test_data=input_data1[training_size:len(input_data1),:]
		#plotting 0th feature with output
		# x1=input_data[:,0]
		# y1=input_data[:,6]

		# print("max t= ",np.amax(y1)," max de= ",np.amax(x1))
		# plt.title("Matplotlib demo") 
		# plt.xlabel("x axis caption") 
		# plt.ylabel("y axis caption") 
		# plt.plot(x1,y1) 			#0- circle, 1- upward parabola, 
		# plt.show()  
		#h(x)=0+01*x0+02*x0+03*x1+
		#training
		degree=2
		theta=np.random.random_sample(size=7)
		# print(theta)

		x = input_data[:,:6]

		y=input_data[:,6]
		theta,mean,std=predict_y(degree,0.1, 10000, theta, x,y)

		#testing
		x1=np.matrix(test_data[:,:6])			#do transpose after it
		x1=x1.astype('float64')
		# # print("y= ",y) 
		x1 = (x1 - mean) / std
		# print(y1)
		prediction=test(degree,x1,theta,mean,std)
		y1=test_data[:,6]
		# print("y,shape= ",y.shape[0]," prediction.shape= ",prediction.shape[1])
		# # print(prediction)
		# plt.title("Matplotlib demo") 
		# plt.xlabel("x axis caption") 
		# plt.ylabel("y axis caption") 
		# plt.plot(y1,prediction.T,"ob") 			#0- circle, 1- upward parabola, 
		# x2=y
		# plt.plot(y,y,label="45deg")
		# plt.show() 









































		# theta=np.matrix([[20],[20],[20],[20],[20],[20],[20],[20],[20],[20],[20],[20],[20]])
		# theta=theta.astype('float64') 
		# print("theta.shape= ",len(theta))
		# # print(theta-((0.1)/len(input_data))*(-560))
		
		# for j in range((input_data).shape[1]-1):
		# 	#Cost Function
		# 	cost_function=0
		# 	for row in range(len(input_data)):
		# 		x=np.matrix(input_data[row][0:len(input_data[0])-1]).transpose()			#do transpose after it
		# 		x=x.astype('float64') 

		# 		x1=[[1],[x[0,0]],[x[0,0]*x[0,0]],[x[1,0]],[x[1,0]*x[1,0]],[x[2,0]],[x[2,0]*x[2,0]], [x[1,0]],[x[3,0]*x[3,0]], [x[4,0]],[x[4,0]*x[4,0]], [x[5,0]],[x[5,0]*x[5,0]]]
		# 		# if(row==0 and j==0):
		# 		# 	print("x1= ",x1)
		# 		#normalization
		# 		for coloumn in range(len(input_data[0])-1):
		# 			mean=np.sum(input_data[:,coloumn])/len(input_data)
		# 			max1=np.amax(input_data[:,coloumn])
		# 			min1=np.amin(input_data[:,coloumn])
		# 			new=((x[coloumn, 0]-mean)/(max1-min1))
		# 			x[coloumn,0]=new
		# 			# print("now x[0][coloumn]= ",x[coloumn,0])

		# 		function_value=h(theta,x)
		# 		# print("function_value= ",function_value)
		# 		cost_function=((function_value-input_data[row][(input_data).shape[1]-1])*(x[j]))[0,0] 
		# 		# print("cost_function ",cost_function,"function_value= ",function_value," actual y= ",input_data[row][(input_data).shape[1]-1]," x[j]= ",x[j])
		# 		theta[j]=theta[j]-((0.1)/len(input_data))*cost_function
		# 		# if(row==0 and j==0):
		# 			# print("function_value= ",function_value," input_data[row][(input_data).shape[1]-1])= ",input_data[row][(input_data).shape[1]-1],"x[j]= ",x[j]," cost_function= ",cost_function)
		# 			# print("after theta= ",theta)
		# 			# print("x= ",x)

		# 		# 	print((theta.transpose())@x)
		# 			# print("adsf")
		# 			# print(theta)

		# 	# dummy=0
		# 	# for wi in range(len(theta)-1):
		# 	# 	dummy+=theta[wi]*theta[wi]

		# 	# cost_function+= (0.01)*(dummy[0][0])
		# 	# theta[j]=theta[j]-((0.1)/len(input_data))*cost_function
		# 	#Gradient descent
		# 	# print("cost_function= ",cost_function)
		# 	# print(((0.1)/len(input_data))*cost_function)
		# 	# theta=theta-((0.1)/len(input_data))*cost_function
		# 	#TILL THIS POINT WE GOT THETA 
		# 	#NOW PREDICT FOR TEST CASES
		# 	#i.e. calculate accuracy

		# print("theta= ",theta)
		# # for row in range(len(input_data)):
		# # 	print(input_data[row])
		# y=np.matrix(input_data[0][0:len(input_data[0])-1]).transpose()			#do transpose after it
		# y=y.astype('float64') 
		# # print("before: \n",y)
		# # # print(x[0][0])

		# # #normaliation
		# for coloumn in range(len(input_data[0])-1):
		# 	mean=np.sum(input_data[:,coloumn])/len(input_data)
		# 	max1=np.amax(input_data[:,coloumn])
		# 	min1=np.amin(input_data[:,coloumn])
		# 	# print("mean= ",mean," max= ",max1," min= ",min1)
		# 	# print("before x[0][coloumn]= ",x[coloumn,0]," x[coloumn, 0]-mean= ",x[coloumn, 0]-mean," (max1-min1)= ",(max1-min1), " new= ",((x[coloumn, 0]-mean)/(max1-min1)))
		# 	new=((y[coloumn, 0]-mean)/(max1-min1))
		# 	y[coloumn,0]=new
		# # y[3]=y[3]-np.sum(input_data[:,3])/len(input_data)
		# print(theta.transpose()@y)			
		# print((input_data[0,0:(len(input_data[0])-1)]))
		



		