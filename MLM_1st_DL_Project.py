# First Keras Deep Learning project from Machine Learning Mastery
# Input by Ryan L Buchanan on a lonely Friday night 28SUG20

# First neural network with keras tutorial 
import numpy as np
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
dataset = loadtxt("pima-indians-diabetes.csv", delimiter=',')

# Split columns into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]