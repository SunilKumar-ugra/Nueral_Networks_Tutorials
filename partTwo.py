import numpy as np 
import matplotlib.pyplot as plt 
import time

X = np.array(([3,5],[5,1],[10,2]),dtype= float)
y = np.array(([80],[75],[56]),dtype = float)

X = X/ np.amax(X, axis = 0)
y = y/100

class Neural_Network(object):
    def __init__(self):
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1
        self.W1 = np.random.rand(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.rand(self.hiddenLayerSize,self.outputLayerSize)
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def forward(self,X):
        self.z2 = np.dot(X,self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        #print (self.z3)
        yHat = self.sigmoid(self.z3)
        return yHat
    def sigmoidPrime(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        #print (self.z3)
        cost = 0.5*((y-self.yHat)**2)
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return cost, dJdW1, dJdW2
    
NN = Neural_Network()
cost1,dJdW1, dJdW2 = NN.costFunctionPrime(X,y)
scalar = 10
a = scalar*dJdW2

NN.W1 = NN.W1 - (scalar*dJdW1)
NN.W2 = NN.W2 - (scalar*dJdW2)
cost2,dJdW2, dJdW1 = NN.costFunctionPrime(X,y)
print (cost1)
print (cost2)

