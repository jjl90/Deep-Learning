import collections
import random as r
import time

import matplotlib.pyplot as plt
import numpy as np
import skimage.color as sc

#inputs
#   NxD matrix x of features
#   MxD matrix W1 of weights between first and second layer
#   1XM matrix W2 of weights between second and third layer
#outputs
#   Nx1 vector y_pred containing the outputs at the last layers
#   NxM matrix z containing the activations for all M hidden neurons

def forward(X, W1, W2):

    N = X.shape[0] #samples
    M = W1.shape[0] #hidden nuerons

    try:
        X.shape[1]
    except:
        N = 1

    if(N > 1):
        D = X.shape[1] #feature dimension

        a = np.zeros((N,M)) #activations before tanh
        z = np.zeros((N,M)) #activations 
        y_pred = np.zeros((N,1))#outputs

        for j in range(0,M):
            a[j] = np.dot(W1,np.transpose(X[j]))

        z = np.tanh(a)

        for j in range(0,N):
            y_pred[j][0] = np.dot(W2, np.transpose(z[j]))

    else:
        D = len(X)
        a = np.zeros((N,M)) #activations before tanh
        z = np.zeros((N,M)) #activations 
        y_pred = np.zeros(N)#outputs

        for j in range(0,M):
            for i in range(0,D):
                a[0][j] = a[0][j] + np.multiply(W1[j][i],X[i])

        z = np.tanh(a)
        y_pred = np.dot(W2, np.transpose(z))

    return y_pred, z


#inputs:
#   NxD matrix X of features
#   Nx1 matrix vector y of ground-truth labels
#   scalar M containing the number of neurons to use
#   scalar iters defining how many iterations
#   scalar eta defining learning rate
#outputs:
#   W1 and W2 defined by forward
#   error_over_times (iters x 1) that contains the error on the sample used in each iteration

def backwards(X, Y, M, iters, eta):

    N = X.shape[0] #samples
    D = X.shape[1] #feature dimension
    error_over_time = np.zeros((iters,1))
    
    #creation/init of weight matricies
    W1 = generate_random_numbers(M,D)
    W2 = generate_random_numbers(1,M)

    for i in range(0,iters): 
        #calculate X, W1, W2 
        randInt = np.random.randint(N)
        y_pred,z = forward(X[randInt], W1, W2) 
        error_over_time[i] = np.power((y_pred - y[randInt]), 2)

        #compute error
        deltaK = np.subtract(y_pred, Y[randInt]) 
        hPrime = 1 - np.power(z,2)
        deltaJ_sum = sum(W2,deltaK)
        deltaJ = np.multiply(hPrime,deltaJ_sum)

        #update weights
        W2 = W2 - eta * deltaK * z
        deltaJ = np.reshape(deltaJ, (deltaJ.shape[1],1))
        W1 = W1 - eta * deltaJ * np.reshape(X[randInt], (1,12)) #30 x 1 and 1 x 12
        
    return W1,W2,error_over_time

#generate random matrix
#i x j matrix
def generate_random_numbers(num_rows, num_cols, mean=0.0, std=0.01):

  ret = np.random.normal(mean, std, (num_rows, num_cols))
  # All your changes should be above this line.
  return ret

if __name__ == "__main__":

    # test data for forward
    # x = np.array([ [1, 2, 3], [4, 5, 6], [7,8,9]]) 
    # y = np.array([[1],[2],[3]])
    # m = 3
    # iters = 2
    # eta = .3
    # W1, W2, error_over_time = backwards(x,y,m,iters,eta)
    # print("W1: ",W1)
    # print("W2: ",W2)
    # print("Error: ",error_over_time)

    #Split Data
    wines = np.genfromtxt("winequality-red.csv", delimiter=";", skip_header=1)
    training_set = np.array_split(wines,2)[0]
    test_set = np.array_split(wines,2)[1]
    ground_truth = training_set[:,training_set.shape[1]-1] #1 Dimensional Array    
    training_set = np.delete(training_set, 11, 1)

    #Standardize Data
    training_mean = np.mean(training_set, axis=0)
    training_std = np.std(training_set,axis=0)

    training_set = training_set - training_mean
    training_set = np.divide(training_set, training_std)
    training_set = np.insert(training_set, training_set.shape[1], 1, axis=1)
    
    #set input variables
    M = 30  #hidden neurons
    iters = 1000 #iterantions
    eta = 0.001
    y = np.reshape(ground_truth,(ground_truth.shape[0],1))
    W1, W2, error_over_time = backwards(training_set, y ,M,iters,eta)
    plt.plot(error_over_time)
    plt.savefig('error_over_time.png')

    y_pred , z = forward(training_set, W1, W2)

    rms = np.sqrt(((y_pred - y)**2).mean())
    print(rms)