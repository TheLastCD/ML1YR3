# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# scikit
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder



def ANN_classifier(epochs,num_neuron,x_train,y_train,x_test,y_test):
    # creating the classifier
    clf = MLPClassifier(hidden_layer_sizes=(num_neuron,num_neuron),activation='logistic',
                        max_iter=epochs)
    
    #training and predicting with classifier
    clf_trained = clf.fit(x_train,y_train)
    clf_predict = clf_trained.predict(x_test)
    
    #accuracy score 
    clf_score = accuracy_score(y_test, clf_predict)
    return clf_score


def Forrest_classifier(num_trees,min_leaf,x_train,y_train,x_test,y_test):
    # creating the classifier
    clf = RandomForestClassifier(n_estimators=num_trees,min_samples_leaf=min_leaf)
    
    #training and predicting with classifier
    clf_trained = clf.fit(x_train,y_train)
    clf_predict = clf_trained.predict(x_test)
    
    #accuracy score
    clf_score = accuracy_score(y_test, clf_predict)
    return clf_score



#task 3.1 ---------------
#importing data set
hiv_data = pd.read_csv('Task3 - dataset - HIV RVG.csv')

#out put min and mean values
print('Whole data description')
print(hiv_data.describe().to_string())

control_data = hiv_data[hiv_data['Participant Condition'] == 'Control']
patient_data = hiv_data[hiv_data['Participant Condition'] == 'Patient']
print('\nControl data description')
print(control_data.describe().to_string())
print('\nPatient data description')
print(patient_data.describe().to_string())


#boxplot foe alpha
plt.figure(figsize=(8,10))
sns.boxplot(data=hiv_data,x='Participant Condition',y='Alpha')
plt.title('boxplots of alpha for each condition')



# density plot foe beta
plt.figure(figsize=(16,8))
status = ['Control','Patient']
for condition in status:
    subset = hiv_data[hiv_data['Participant Condition'] == condition]

    sns.kdeplot(subset['Beta'],label = condition)

plt.legend(prop={'size': 16}, title = 'Condition')
plt.title('density plot of beta for each condition')
plt.xlabel('Beta (min)')
plt.ylabel('Density')



# task 3.2 ------------------------

#splitting the data
train,test = train_test_split(hiv_data,test_size=0.1,shuffle = True)

#getting test and trained values 
x_train = train.iloc[:,3:8].values
y_train = train.iloc[:,8].values

x_test = test.iloc[:,3:8].values
y_test = test.iloc[:,8].values

#ANN ----------------
#ANN_classifier = MLPClassifier(hidden_layer_sizes=(500,500),activation='logistic')

#score for basic version
print('test ann score: ',ANN_classifier(100, 500, x_train, y_train, x_test, y_test))

#loop the number of max itteration/epochs
epochs = [50,100,200,400,500,700,800,1000,2000,2500]
acc_score = []
for num in epochs:
    ann_score= ANN_classifier(num, 500, x_train, y_train, x_test, y_test)
    acc_score.append(ann_score)

#graph epochs to accuracy
plt.figure(figsize=(16,10))
plt.plot(epochs,acc_score,'b-')

#tree classifiers
f_score= Forrest_classifier(1000, 5, x_train, y_train, x_test, y_test)
print('\nleaf samples = 5:',f_score)
f_score= Forrest_classifier(1000, 10, x_train, y_train, x_test, y_test)
print('\nleaf samples = 10: ',f_score)


#task 3.3 cross validation ---------------------------
#values to test
ann_num = [50,500,1000]
tree_num = [50,500,10000]

#splitting the data
#x_data = hiv_data.iloc[:,1:8].values
x_data = hiv_data.iloc[:,3:8].values
y_data = hiv_data.iloc[:,8].values

#scaling the data
scaler = MinMaxScaler()
scaler = scaler.fit(x_data)
scaled_x_data = scaler.transform(x_data)

le = LabelEncoder()
le = le.fit(y_data)
encoded_y_data = le.transform(y_data)

#array holding the 2 data sets 
datasets = [x_data,scaled_x_data]

clf_type_cv_score = []
for data in datasets:
    print('data')
    labels = ['50','500','1000/10000']
    for num in range(len(ann_num)):
        print(num)
        #classifiers
        ann_clf = MLPClassifier(hidden_layer_sizes=(ann_num[num],ann_num[num]),activation='logistic')
        tree_clf = RandomForestClassifier(n_estimators=tree_num[num],min_samples_leaf=10)
        
        #cross validation score generator
        ann_score = (cross_val_score(ann_clf, data,encoded_y_data,cv=10)).mean()
        tree_score = (cross_val_score(tree_clf, data,encoded_y_data,cv=10)).mean()
        
        clf_type_cv_score.append([labels[num],ann_score,tree_score])

    #creating the pd data frame for displaying data
    score = np.array(clf_type_cv_score)
    score_df = pd.DataFrame(score, columns=['label','ANN','random forrest'])
    print(score_df.to_string())
    
    clf_type_cv_score = []


"""
ann_clf = MLPClassifier(hidden_layer_sizes=(500,500),max_iter=10000,random_state=(1235),activation='logistic',solver='adam')
tree_clf = RandomForestClassifier(n_estimators=1000,min_samples_leaf=10)

ann_score1 = (cross_val_score(ann_clf, scaled_x_data,y_data,cv=10)).mean()
ann_score2 = (cross_val_score(ann_clf, scaled_x_data,encode_y_data,cv=10)).mean()

tree_score1 = (cross_val_score(tree_clf, scaled_x_data,y_data,cv=10)).mean()
tree_score2 = (cross_val_score(tree_clf, scaled_x_data,encode_y_data,cv=10)).mean()

print("ann")
print('not scaled',ann_score1)
print('scaled',ann_score2)
print("tree")
print('not scaled',tree_score1)
print('scaled',tree_score2)
"""
