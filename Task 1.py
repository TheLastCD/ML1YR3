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

# x_train, x_test, y_train, y_test = train_test_split(data['x'], data['y'], test_size=0.3, shuffle = False)

x_train = x_train.sort_values()
y_train = y_train.sort_values()
x_test = x_test.sort_values()
y_test = y_test.sort_values()



def getPolynomialDataMatrix(x, degree):
    # x: training x values (Numpy 1D array)
    # degree: polynomial degree integer (integer)
    # x = x[:-12]
    #Creates new array degExponent containing only 1s
    degExponent =  np.ones(x.shape)
    
    #creates an degree number of columns each containing the exponent of x uptil degree
    for i in range(1,degree + 1):
        degExponent= np.column_stack((degExponent, x ** i))
        
    # print(np.delete(degExponent,0,1))
    # input()
    # return exponent array with delete first column of ones
    return np.delete(degExponent,0,1)


def getWeightsForPolynomialFit(x,y,degree):
    # x: training x values (Numpy 1D array)
    # y: training y values (Numpy 1D array)
    # degree: polynomial degree (integer)
    
    # runs getPolynomialDataMatrixfunction
    X = getPolynomialDataMatrix(x, degree)

    # transpose and dot product against original matrix 
    XX = X.transpose().dot(X)
    # solves the matrix 
    w = np.linalg.solve(XX, X.transpose().dot(y))
    #w = np.linalg.inv(XX).dot(X.transpose().dot(y))
    
    # returns the solved matric
    return w

# so to get training data to plot needed need to add an arguement that makes it plot accurately 
def pol_regression(features_train, y_train,degree):
    # features_train: x values (Numpy 1D array)
    # y_train: y values (Numpy 1D array)
    # degree: polynomial degree(integer)
    
    # a 0 degree polynomial is a constant value
    # this calculates the mean of the data as that is an approximation of all the data points
    # http://polynomialregression.drque.net/math.html
    # https://towardsdatascience.com/polynomial-regression-the-only-introduction-youll-need-49a6fb2b86de
    if(degree == 0):
        # Calculates the mean of the whole training set
        parameters = np.repeat(np.mean(y_train),features_train.size)
        # XX = features_train.transpose().dot(features_train)
        # parameters = y_train/XX


    else:
        #receives the solved matrix
        w1 = getWeightsForPolynomialFit(features_train,y_train,degree)
        # creates a second exponent matrix this time for fitting the data
        Xtest1 = getPolynomialDataMatrix(features_train, degree)
        # performs the dot product of the solved matrix and the new exponent matrix
        parameters = Xtest1.dot(w1)
    #return result 
    return parameters




#modified so testing data can be inputted 
def pol_regression_test(features_train, y_train,degree,test):
    # features_train: x values (Numpy 1D array)
    # y_train: y values (Numpy 1D array)
    # degree: polynomial degree(integer)
    
    # a 0 degree polynomial is a constant value
    # this calculates the mean of the data as that is an approximation of all the data points
    # http://polynomialregression.drque.net/math.html
    # https://towardsdatascience.com/polynomial-regression-the-only-introduction-youll-need-49a6fb2b86de
    if(degree == 0):
        ytest1 = np.repeat(np.mean(y_train),test.size)
        # XX = features_train.transpose().dot(features_train)
        # ytest1 = y_train/XX
        
    else:
        w1 = getWeightsForPolynomialFit(features_train,y_train,degree)
        Xtest1 = getPolynomialDataMatrix(test, degree)
        ytest1 = Xtest1.dot(w1)
    return ytest1

def eval_pol_regression(parameters, x, y, degree):
    # Parameters: return of pol_regression function (Numpy 1D array)
    # x: x values (Numpy 1D array)
    # y: y values (Numpy 1D array)
    # degree: polynomial degree (Integer)

    mse = np.square(np.subtract(parameters,y)).mean() 
    rmse = math.sqrt(mse)
    return rmse


    
    



# 0 no one has any clue what 0 degree should result in
# 0 should maybe return the mean average of the training data
# Source: http://polynomialregression.drque.net/math.html
#alex went with x+y
#np.linspace
degs = [0,1, 2, 3, 6, 10]




# rmsearrt= []
# for i in degs:
#     print(eval_pol_regression(pol_regression_test(x_train,y_train,i, y_test), x_test,y_test,i))  
#     rmsearrt.append(eval_pol_regression(pol_regression_test(x_train,y_train,i,x_test),x_test,y_test,i))


# plt.figure()
# plt.plot(degs,rmsearrt, 'b')
# plt.plot
# plt.show();


    

# for i in degs:
#     title = "Degrees: "+ str(i)
#     plt.figure(figsize=(8, 6), dpi=100)
#     plt.title(title)
#     plt.ylabel('X ')
#     plt.xlabel('Feature')
#     plt.xlim(-5,5)
#     plt.plot(x_train,y_train, 'bo')
#     # plt.plot(x_test,y_test, 'go')
#     plt.plot(x_train, pol_regression(x_train,y_train,i), 'r')

pol_return = []
for i in degs:
    pol_return.append(pol_regression(x_train,y_train,i))

title = "plotted polynomials"
plt.figure(figsize=(8, 6), dpi=100)
plt.title(title)
plt.ylabel('X ')
plt.xlabel('Feature')
plt.xlim(-5,5)

plt.plot(x_test,y_test, 'go')
deg_0 = plt.plot(x_train, pol_return[0], 'r', label='0')
deg_1 = plt.plot(x_train, pol_return[1], 'g', label='1')
deg_2 = plt.plot(x_train, pol_return[2], 'b', label='2')
deg_3 = plt.plot(x_train, pol_return[3], 'y', label='3')
deg_4 = plt.plot(x_train, pol_return[4], 'c', label='6')
deg_5 = plt.plot(x_train, pol_return[5], 'tab:orange', label='10')
plt.plot(x_train,y_train, 'mo')
plt.legend(['0','1','2','3','6','10'])


