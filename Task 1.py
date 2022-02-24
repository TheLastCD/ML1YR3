import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import numpy.linalg as linalg
import math

data = pd.read_csv('Task1 - dataset - pol_regression.csv')

x_train = data['x']
y_train = data['y']

x_test = data['x']
y_test = data['y']


x_train = x_train.sort_values()
y_train = y_train.sort_values()
x_test = x_test.sort_values()
y_test = y_test.sort_values()




def getPolynomialDataMatrix(x, degree): #Feature Expansion
    # x: training x values (Numpy 1D array)
    # degree: polynomial degree integer (integer)
    # x = x[:-12]
    #Creates new array degExponent containing only 1s
    degExponent =  np.ones(x.shape)
    
    #creates an degree number of columns each containing the exponent of x uptil degree
    for i in range(1,degree + 1):
        degExponent= np.column_stack((degExponent, x ** i))
    # return exponent array and delete first column of ones
    
    # return np.delete(degExponent,0,1)
    return degExponent



# so to get training data to plot needed need to add an arguement that makes it plot accurately 
def pol_regression(features_train, y_train,degree):
    # features_train: x values (Numpy 1D array)
    # y_train: y values (Numpy 1D array)
    # degree: polynomial degree(integer)
    
    # a 0 degree polynomial is a constant value
    # this calculates the mean of the data as that is an approximation of all the data points
    if(degree == 0):
        # Calculates the mean of the whole training set and repeats it for the length y
        parameters = np.repeat(np.mean(y_train),features_train.size)
        
        return parameters


    else:
        X = getPolynomialDataMatrix(features_train, degree)
        # transpose and dot product against original matrix 
        XX = X.transpose().dot(X)
        Y= X.transpose().dot(y_train)

        # solves the matrix 
        #inverts XX then dots it with Y producing the alpha coeffeicients
        return linalg.solve(XX,Y)


def eval_pol_regression(parameters, x, y, degree):
    # Parameters: return of pol_regression function (Numpy 1D array)
    # x: x values (Numpy 1D array)
    # y: y values (Numpy 1D array)
    # degree: polynomial degree (Integer)    
    testMatrix = getPolynomialDataMatrix(x,degree)

    if degree == 0:
        predictedy= testMatrix
    else:
        predictedy = testMatrix.dot(parameters)
    
    mse = np.mean((predictedy-y)**2)
    rmse = math.sqrt(mse)
    return rmse
 


degs = [0,1,2, 3, 6, 10]

pol_return = []


for i in degs:
    output = pol_regression(x_train,y_train,i)
    l = getPolynomialDataMatrix(x_train,i)
    if i == 0:
        l = output
    else:
        l = l.dot(output)
    pol_return.append(l)


title = "plotted polynomials"
plt.figure(figsize=(8, 6), dpi=100)
plt.title(title)
plt.ylabel('Y')
plt.xlabel('X')
plt.xlim(-5,5)
plt.plot(x_test,y_test, 'go')
deg_0 = plt.plot(x_train, pol_return[0], 'r', label='0')
deg_1 = plt.plot(x_train, pol_return[1], 'g', label='1')
deg_2 = plt.plot(x_train, pol_return[2], 'b', label='2')
deg_3 = plt.plot(x_train, pol_return[3], 'y', label='3')
deg_4 = plt.plot(x_train, pol_return[4], 'c', label='6')
deg_5 = plt.plot(x_train, pol_return[5], 'tab:orange', label='10')
plt.plot(x_train,y_train, 'mo')
plt.legend(['data','0','1','2','3','6','10'])

x_train, x_test, y_train, y_test = train_test_split(data['x'], data['y'], test_size=0.3, shuffle = True)


test_rmse = []
train_rmse = []
for i in degs:
    output = pol_regression(x_train,y_train,i)
    pol_return.append(output)
    test_rmse.append(eval_pol_regression(output,x_test,y_test,i))
    train_rmse.append(eval_pol_regression(output,x_train,y_train,i))




title = "Rmse"
plt.figure(figsize = (8,6), dpi = 100)
plt.ylabel('Error')
plt.xlabel('Degree')
test = plt.plot(degs,test_rmse, 'r')
train = plt.plot(degs,train_rmse, 'b')
plt.legend(["Testing Data","Training Data"])



