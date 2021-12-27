import pandas as pd
import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics

data = pd.read_csv('Task3 - dataset - HIV RVG.csv')

train = data.sample(frac = 0.9)
test = data.drop(train.index)
Alpha_train = train['Alpha'].values
Beta_train = train['Beta'].values
Lambda_train = train['Lambda'].values
Lambda1_train = train['Lambda1'].values
Lambda2_train = train['Lambda2'].values
Condition = train['Participant Condition'].values
Bifurcation_train = train['Bifurcation number'].values

def RDC(X_train, Y_train, trees, samples):
    plot_step = 0.02
    clf = RandomForestClassifier(n_estimators= trees, min_samples_leaf= samples, max_depth=None)
    X_train = X_train.reshape(-1, 1)
    # Y_train = Y_train.reshape(-1, 1)
    
    # put your code here to fit the model you have declared above
    decForest = clf.fit(X_train, Y_train)
    
    y_predict = decForest.predict(X_train)
    
    x_min, x_max = X_train.min() - .5, X_train.max() + .5
    y_min, y_max = Y_train.min() - .5, Y_train.max() + .5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                              np.arange(y_min, y_max, plot_step))
    
    plt.figure()
    xl = np.c_[xx.ravel(),yy.ravel()]
    # print(xl[:,:2])
    Z = clf.predict(np.c_[xl[:,:0], xl[:,:1]])
    Z = Z.reshape(xx.shape)
    
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    
    plt.scatter(X_train[:, 0], X_train[:, 0], c=Y_train, cmap=plt.cm.tab20c)
    
    # plt.xlabel(iris.feature_names[0])
    # plt.ylabel(iris.feature_names[1])
    # plt.axis("tight")
    # plt.savefig('classification_forest.png')
    # plt.show()
    
RDC(Alpha_train,Bifurcation_train,1000,10)
