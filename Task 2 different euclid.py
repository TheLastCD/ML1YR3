import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import numpy.linalg as linalg
import random


data = pd.read_csv('Task2 - dataset - dog_breeds.csv')


def compute_euclidean_distance(vec_1, vec_2):
    # vec_1 , vec_2 two vectors and it calculates the distance
    distance = np.sum(np.square(vec_1 - vec_2)) 
    return np.sqrt(distance)

def initialise_centroids(dataset, k): #k = 2,3 
    # dataset: the data set unsplit
    # k: number of classifiers 
    # initialise a numpy array of just ones the shape of the data
    centroids = np.ones(dataset.shape[1])
    
    # generate random centroid points with the data range
    for i in range(0,k):
        # empty array for random values
        new = []
        # cycle through all features
        for j in range(0,dataset.shape[1]):   
            # create a random value within the range of the data and appened it to the new array
            new.append(random.uniform(min(dataset[:, j]),max(dataset[:,j])))
        # conver the new array to an np array and add it to the end of the new centroid values
        centroids= np.column_stack((centroids, np.array(new)))
    # slice of the ones created at the start and transpose the array so that it matches the axis of the features
    return (np.delete(centroids,0,1)).transpose()

def get_class(dataset,k,centroids):
    # dataset: the data set 
    # k: k value
    # centroid: classification centres
    # if there has already been a classification assigned wipe it 
    if(len(dataset.shape) == 5):
        np.delete(dataset,0,5)
    # initialize empty array for holding classification data
    classification = []
    # loop through the dataset

    for i in dataset:
        # initialize empty array to temporarily hold the distances
        distances = []
        # loop through k
        for dist in range(0,k):
            # calculate euclidean distance between point and cluster centre
            distances.append(compute_euclidean_distance(centroids[dist],i))
        # append the shortest the classified distance to the classification array
        classification.append(distances.index(min(distances)))   
    # return the classification array merged with dataset array adding the classification
    return np.column_stack((dataset,np.array(classification)))

def new_cent(classified_set,k):
    # initialise an a numpy array of just ones
    centroids = np.ones((classified_set.shape[1]-1))
    # loop through k
    for fil in range(0,k):
        # use the i value to filter all the different k classifications and add them to a temporary array
        filtered = classified_set[classified_set[:,4] == fil]
        # empty array for holding the new centroid locations
        new = []
        # calculate and save the mean of these columns
        for i in range(classified_set.shape[1]-1):
            new.append(np.mean(filtered[:,i]))
        # append to the centroids array
        centroids= np.column_stack((centroids,np.array(new)))
    # return the centroid array with the 1s removed
    return(np.delete(centroids,0,1)).transpose()

def kmeans(dataset, k):
    # get the random initial centroids
    centroids = initialise_centroids(dataset, k) 
    # set the number of iterations
    iterations = 1000000
    tempcentroids = np.ones(centroids.shape)
    # iterate through the iterations
    cluscatmap = []
    for i in range(iterations):
        # run classificiation
        cluster_assigned= get_class(dataset,k,centroids)
        # calculate new centroids
        centroids = new_cent(cluster_assigned,k)
        if((centroids == tempcentroids).all()):
            break
        else:
            tempcentroids = centroids

        cluscatmap.append([i, within_cluster_scatter(cluster_assigned,centroids,k)])
    # return the final centroid location and the data with attached classification and the withiin cluster scatter map
    return centroids, cluster_assigned, np.array(cluscatmap)

def within_cluster_scatter(dataset,centroids,k):
    sum_square_euclid = 0
    for i in range(0,k):
        for j in dataset[dataset[:,4] == i]:            
            sum_square_euclid = sum_square_euclid + (compute_euclidean_distance(centroids[i], j[0:4])**2)
    return sum_square_euclid

            
        



# k = 2 plotting 
centroid_marker, classified, iteration_scatter = kmeans(data.to_numpy(),2) #k=2,3
zeros = classified[classified[:,4] == 0]
ones = classified[classified[:,4] == 1]

# height and tail length
plt.figure(figsize=(5, 3), dpi=1000)
plt.scatter(zeros[:,0],zeros[:,1], s = 6)
plt.scatter(ones[:,0], ones[:,1], s= 6)
plt.scatter(centroid_marker[:,1],centroid_marker[:,1], s= 10, marker = '*', c = 'crimson')
plt.title('K = 2')
plt.ylabel('Tail Length')
plt.xlabel('Height')
plt.legend(['1', '2','Centroids'])
plt.show()

# height and leg length
plt.figure(figsize=(5, 3), dpi=1000)
plt.scatter(zeros[:,0],zeros[:,2], s = 6)
plt.scatter(ones[:,0], ones[:,2], s= 6)
plt.scatter(centroid_marker[:,0],centroid_marker[:,2], s= 10, marker = '*', c = 'crimson')
plt.title('K = 2')
plt.ylabel('Leg Length')
plt.xlabel('Height')
plt.legend(['1', '2','Centroids'])
plt.show()

plt.figure(figsize=(8, 6), dpi=300)
plt.xlim(0.5,iteration_scatter.shape[0]+0.5)
iterations = plt.plot(iteration_scatter[:,0]+1,iteration_scatter[:,1])
plt.title("Within cluster scatter over iterations")
plt.xlabel("Iteration Step")
plt.ylabel("Within Cluster Scatter")


# k = 3 plotting 
centroid_marker, classified, iteration_scatter = kmeans(data.to_numpy(),3) #k=2,3

zeros = classified[classified[:,4] == 0]
ones = classified[classified[:,4] == 1]
twos = classified[classified[:,4] == 2]

# height and tail length
plt.figure(figsize=(5, 3), dpi=1000)
plt.scatter(zeros[:,0],zeros[:,1], s = 6)
plt.scatter(ones[:,0], ones[:,1], s= 6)
plt.scatter(twos[:,0], twos[:,1], s= 6)
plt.scatter(centroid_marker[:,0],centroid_marker[:,1], s= 10, marker = '*', c = 'crimson')
plt.title('K = 3')
plt.ylabel('Tail Length')
plt.xlabel('Height')
plt.legend(['1', '2','3','Centroids'])
plt.show()

# height and leg length
plt.figure(figsize=(5, 3), dpi=1000)
plt.scatter(zeros[:,0],zeros[:,2], s = 6)
plt.scatter(ones[:,0], ones[:,2], s= 6)
plt.scatter(twos[:,0], twos[:,2], s= 6)
plt.scatter(centroid_marker[:,0],centroid_marker[:,1], s= 10, marker = '*', c = 'crimson')
plt.title('K = 3')
plt.ylabel('Leg Length')
plt.xlabel('Height')
plt.legend(['1', '2','3','Centroids'])
plt.show()

plt.figure(figsize=(8, 6), dpi=300)
plt.xlim(0.5,iteration_scatter.shape[0]+0.5)
iterations = plt.plot(iteration_scatter[:,0]+1,iteration_scatter[:,1])
plt.title("Within cluster scatter over iterations")
plt.xlabel("Iteration Step")
plt.ylabel("Within Cluster Scatter")








