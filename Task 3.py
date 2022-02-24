# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, normalize, LabelEncoder
import dataframe_image as dfi
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score



data = pd.read_csv('Task3 - dataset - HIV RVG.csv')



def Stats(frame,statusCategory,columns):
    #get all rows with the assigned status
    fullData= []
    df2 = frame.loc[frame['Participant Condition'] == statusCategory]
    df2 = df2.iloc[:,3:8]


    #Find maximum, minimum, mode, median, mean, and variance of the data
    fullData.append(df2.max().tolist())
    fullData.append(df2.min().tolist())
    fullData.append(df2.median().tolist())
    fullData.append(df2.mean().tolist())
    fullData.append(df2.mode().values[0].tolist())
    fullData.append(df2.var().tolist())
    fullData.append(np.sqrt(df2.var().tolist()))
    fullData = np.array(fullData).transpose()

    # Stack this into a pandas dataframe for ease of reading
    StatFrame = pd.DataFrame(fullData,columns=['max', 'min','median','mean','mode','var','stand dev'],index = columns[3:8])
    dfi.export(StatFrame,str(statusCategory+ ".png") )
    return StatFrame

def GenPlot(data,columnNames):
    # get patient and control frames
    Patient = Stats(data,"Patient",columnNames)
    Control = Stats(data,"Control",columnNames)
    # Show them to the user
    print("Patient Statistics")
    print(Patient)
    print("Control Statistics")
    print(Control)
    
    # Plot Box Plot Alphas
    Box = [Patient.values.tolist()[0][0:3],Control.values.tolist()[0][0:3]]
    
    fig1, ax1 = plt.subplots(figsize = (10,8),dpi = 1000)
    ax1.set_title('Alpha for both Categories')
    ax1.set_xticklabels(["Patient", "Control"], ha="center")
    ax1.boxplot(Box)
    plt.show()
    
    # Plot Density of Betas
    sns.set(style="darkgrid")
    
    sns.kdeplot(data.loc[data['Participant Condition'] == "Patient"]['Beta'])
    sns.kdeplot(data.loc[data['Participant Condition'] == "Control"]['Beta'])
    plt.title('Density Plot of Betas')
    plt.legend(['Patient','Control'])
    plt.show()


def DataNormalizer(data):
    # Normalise the data so that it falls between 0 and 2
    scaler = MinMaxScaler(feature_range = (0,2))
    scaler = scaler.fit(data)
    return scaler.transform(data)


def DataSplit(data,columnNames,columnStart):
    # Split the data so that Alpha ,Beta and the two lambdas are used as x values and the participant condition as th y
    x = DataNormalizer(data.iloc[:,columnStart:8]) 
    y = data.iloc[:,8].values
    xTrain,xTest,yTrain,yTest = train_test_split(x,y, test_size=0.1, shuffle = True)
    # show the split
    unique, counts = np.unique(yTest, return_counts=True)
    print("Testing Set: ")
    print(dict(zip(unique, counts)))
    
    unique, counts = np.unique(yTrain, return_counts=True)
    print("Training Set: ")
    print(dict(zip(unique, counts)))

    return xTrain,xTest,yTrain,yTest


def ANN(epoch, neurons):
    # setup  an ANN using logistic activation function, adamn solver, with user set epochs and neurons
    print("Number of epochs:  " , str(epoch))
    print("Number of neurons: ", str(neurons) )
    clf = MLPClassifier(max_iter=epoch, activation = 'logistic'
                        ,hidden_layer_sizes=(neurons[0], neurons[1]), solver = "adam")
    return clf

    
    
    
def RFC(samples, estimators):
    # setup and RFC with user se samples and estimators 
    print("Number of Samples: " + str(samples))
    print("Number of Trees: "+ str(estimators))
    clf = RandomForestClassifier(n_estimators = estimators,min_samples_leaf = samples)
    
    return clf


def ClassEval(clf,xTrain,xTest,yTrain,yTest):
    # Fit the classifier to the data
    clf.fit(xTrain, yTrain)
    # predict the classifier
    predictions =clf.predict(xTest)
    # show classified stats
    unique, counts = np.unique(predictions, return_counts=True)
    print("Testing Set: ")
    print(dict(zip(unique, counts)))
    
    # do the same on the training data
    predictions =clf.predict(xTrain)
    unique, counts = np.unique(predictions, return_counts=True)
    print("Training Set: ")
    print(dict(zip(unique, counts)))
    # calculate the accuracy of the classifier
    accScore = accuracy_score(yTest, clf.predict(xTest))
    print("Accuracy Score: " + str(accScore))
    print('\n')
    return accScore

def CrossVal(xTest,yTest,NeuNet,DecFor,cross):
    # cross eval on the neural network and the random forest classifier
    Neurscores = cross_val_score(NeuNet, xTest, yTest, cv=cross)
    Treescores = cross_val_score(DecFor, xTest, yTest, cv=cross)
    print("Trees Scores")
    print(Treescores)
    print("ANN Scores")
    print(Neurscores)
    # Calculate the mean and return it as an array
    print("ANN Mean Scores: " + str(Neurscores.mean()))
    print("RFC Mean Scores: " + str(Treescores.mean()))
    return [Neurscores.mean(),Treescores.mean()]

    
# get the column names
columnNames = []
for c in data.columns:
    columnNames.append(c)


# Tasks: 3.1 Get Stats
GenPlot(data,columnNames)


# Tasks 3.2: ANN And Decision Tree implementation

xTrain,xTest,yTrain,yTest = DataSplit(data,columnNames,3)
xTrain1,xTest1,yTrain1,yTest1 = DataSplit(data,columnNames,0)
# See how the iterations effects accuracy
iterationTest = [1,100,250,500,750,1000,2500,5000,10000]
leafs = [5,10]
scoresalpa = []
scoresimage = []
for i in iterationTest:
    NN = ANN(i,[500,500])
    scoresalpa.append(ClassEval(NN,xTrain,xTest,yTrain,yTest))
    scoresimage.append(ClassEval(NN,xTrain1,xTest1,yTrain1,yTest1))

plt.figure(figsize = (10,8),dpi = 1000)
plt.title("Accuracy as iterations increase")
plt.ylabel("Accuracy")
plt.xlabel("Iterations")
plt.plot(iterationTest,scoresalpa,c= 'b')
plt.plot(iterationTest,scoresimage,c= 'r')
plt.legend(["Alpha...Lamba2", "Image Number... Lambda2"])

# See how leaf nodes effect accuracy
scoresalpa = []
scoresimage = []
for i in leafs:
    FC = RFC(i, 100)
    scoresalpa.append(ClassEval(FC,xTrain,xTest,yTrain,yTest))
    scoresimage.append(ClassEval(FC,xTrain1,xTest1,yTrain1,yTest1))
plt.figure(figsize = (10,8),dpi = 1000)
plt.title("Accuracy as leaf nodes increase")
plt.ylabel("Accuracy")
plt.xlabel("Nodes")
plt.plot(leafs,scoresalpa,c= 'b')
plt.plot(leafs,scoresimage,c= 'r')
plt.legend(["Alpha...Lamba2", "Image Number... Lambda2"])
    

# Task 3.3: Cross validation

# set the iterations and forest size for cross val 
ANNneurons = [50,500,1000]
RFCtrees = [50,500,10000]
scores =[]
# split the data
x = data.iloc[:,0:7].values
y = data.iloc[:,8].values

# run cross val 
for i in range(len(ANNneurons)):
    NeuNet = ANN(100,[ANNneurons[i],ANNneurons[i]])
    DecFor = RFC(10, RFCtrees[i])
    # store cross val scores
    scores.append(CrossVal(x,y,NeuNet,DecFor,10))

# pack scores into pandas dataframe
StatFrame = pd.DataFrame(scores,columns=['ANN', 'RFC'],index = ['50','500','1000/ 10000'])
print(StatFrame)
# dfi.export(StatFrame,"cross Eval results" )

# display scores jin plots
plt.figure(figsize = (10,8),dpi = 1000)
plt.title('Mean of cross evaluation by Epoch')
plt.plot(ANNneurons,np.array(scores).transpose()[0], c = 'b')
plt.xlabel("Neurons")
plt.ylabel("Accuracy")
plt.figure(figsize = (10,8),dpi = 1000)
plt.title('Mean of cross evaluation by Forest, leaf nodes set to 5')
plt.plot(RFCtrees,np.array(scores).transpose()[1], c = 'r')
plt.xlabel("Trees")
plt.ylabel("Accuracy")






