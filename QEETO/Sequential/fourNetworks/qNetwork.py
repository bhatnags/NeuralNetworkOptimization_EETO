import random

from keras.datasets import cifar10, cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

import math
import numpy as np


'''
Contains functions of Network inspired by the concepts of Quantum Mechanics:
	Initializes the network class without any hyper-parameter set
	Orthonormal basis states defining quantum states
	Hadamard Gate for creation of super-position
	Randomize the values in Bloch Sphere
	Get the Quantum Population Vector
	Make the Qunatum Measurement
	Initializes network with randomly selected hyper-parameter
'''

class qgaNetwork():

	'''
    Initializes the network class
        and it's features - number of networks, param, network, quantum population vectors and array size
        without any hyper-parameter set (param is set to none)
    '''
	def __init__(self, numNetworks, numParam, param=None):
		self.numNetworks = numNetworks
		self.param = param
		self.network = {}
		self.numParam = numParam # num of parameters to optimize
		self.top_bottom = 3 # Array to get Quantum population vector
		self.qpv = np.empty([self.numNetworks, self.numParam, self.top_bottom])
		self.nqpv = np.empty([self.numNetworks, self.numParam, self.top_bottom])


	'''
	Orthonormal basis states or basis vectors - ket zero and ket one
	the quantum state of a qubit can be represented as a linear superposition of its two basis vector states

	Qubit zero => ket zero
	|0>  =  [1, 0]

	Qubit one => ket one
	|1>  =  [0, 1]
	'''
	def basisVectors(self):
		ketZero = np.array([[1], [0]])
		ketOne = np.array([[0], [1]])
		return ketZero, ketOne



	'''
	HADAMARD GATE
		takes in ket zero (|0>) or ket one (|1>) 
		and gives ( |0> +/- |1> )/root(2)
		thereby creating  superposition, that the output can take any value between 0/1 
	'''
	def HadamardGate(self):
		# Hadamard gate
		r2=math.sqrt(2.0)           
		h=np.array([[1/r2, 1/r2],[1/r2,-1/r2]])
		return h



	'''
	Get random theta value in the Bloch Sphere
		returns random theta in the Bloch sphere
	'''
	def getRandomTheta(self):
		theta = np.random.uniform(0,1)*90
		theta = math.radians(theta)
		theta = float(theta)
		return theta


	'''
	Get the angles with which the basis vectors are to be rotated
	Uses randomly generated theta
	returns the rotation angles 
	'''
	def getRotationAngles(self, theta):
		theta = float(theta)
		rot1=float(math.cos(theta)); rot2=-float(math.sin(theta));
		rot3=float(math.sin(theta)); rot4=float(math.cos(theta));
		return rot1, rot2, rot3, rot4


	'''
	Get the Quantum Population Vector
	Has the probabilistic details of all the networks and possibility of the hyper-parameters
	'''
	def getPopulationVector(self, AlphaBeta, ketZero, ketOne, h, theta, rot):

		# Values for all the networks
		for i in range(0, self.numNetworks):

			# Values for all the parameters in the network
			for j in range(0, self.numParam):

				# Random rotation
				theta = self.getRandomTheta()
				rot1, rot2, rot3, rot4 = (self.getRotationAngles(theta))
				AlphaBeta[0] = rot1 * (h[0][0] * ketZero[0]) + rot2 * (h[0][1] * ketZero[1])
				AlphaBeta[1] = rot3 * (h[1][0] * ketOne[0]) + rot4 * (h[1][1] * ketOne[1])

				# Probability of getting 0
				# alpha squared
				self.qpv[i, j, 0] = np.around(2 * pow(AlphaBeta[0], 2), 2)

				# Probability of getting 1
				# beta squared
				self.qpv[i, j, 1] = 1 - self.qpv[i, j, 0]

		# Return Quantum Population Vector
		return self.qpv


	'''
	Make the Measurement
		Any quantum state can be represented as a superposition of the eigenstates of an observable
		Measurement results in the system being in the eigenstate corresponding to the eigenvalue result of the measurement
	Every key of the parameters in hyper-parameter set is given a range
	If there are 5 keys, the keys will have ranges: (0,0.2), (0.2+, 0.4), (0.4+, 0.6), (0.6+, 0.8), (0.8+, 1)
	A random real number is generated
	Based on it's proximity with the key's range, the keys are selected, and
	measurement is said to be done
	'''
	def Measure(self, pv):

		# Re-Initialize the network
		self.network = {}

		# For all the networks
		for i in range(self.numNetworks):

			# Initialize the network
			self.network[i] = {}

			# For all the keys of the parameters
			p = 0
			for key in self.param:

				# get number of options in each parameter =>numParamLen
				numParamLen = (len(self.param[key]))

				# Get a random value
				rand = random.randint(0, 1)

				val = pv[i, p, rand]

				# For all the keys in parameters
				for n in range(numParamLen):
					comp1 = n / float(numParamLen)
					comp2 = (n + 1) / float(numParamLen)

					if ((val >= comp1) & (val <= comp2)):
						self.network[i][key] = self.param[key][n+1]
				p=p+1

		# Return the network
		return self.network




	'''
    Initializes network with randomly selected hyper-parameter
        Returns the network
    '''
	def Init_population(self):

		# Get |0> and |1>
		ketZero, ketOne = self.basisVectors()

		# Create an empty numpy array for the probability of 0/1 values
		AlphaBeta = np.empty([self.top_bottom])

		# Get Hadamard Gate operator
		h = self.HadamardGate()
		
		# Rotation Q-gate
		theta = 0
		rot = np.empty([2,2])

		# Initial population array (individual x chromosome)
		i=0; j=0;

		self.qpv = self.getPopulationVector(AlphaBeta, ketZero, ketOne, h, theta, rot)

		self.network = self.Measure(self.qpv)

		return self.qpv, self.network





'''
Class to get the accuracy of the networks
'''

earlyStopper = EarlyStopping( monitor='val_loss', min_delta=0.1, patience=5, verbose=0, mode='auto' )


class fitness():
	'''
    Initializes the fitness class
        and it's features - fitness, number of networks
        without any hyper-parameter set (param is set to none)
    '''

	def __init__(self, numNetworks):
		self.numNetworks = numNetworks
		self.network = {}
		self.fitness = np.empty([self.numNetworks])


	'''
    Get number of classification classes in the dataset
    '''
	def getnbClasses(self, dataset):
		if dataset == 'cifar10':
			nbClasses = 10
		elif dataset == 'cifar100':
			nbClasses = 100
		return nbClasses




	'''
    Get the details of the training and test dataset	
    '''
	def getData(self, dataset, nbClasses):
		if dataset == 'cifar10':
			(x_train, y_train), (x_test, y_test) = cifar10.load_data()
		elif dataset == 'cifar100':
			(x_train, y_train), (x_test, y_test) = cifar100.load_data()

		x_train = x_train.reshape(50000, 3072)
		x_test = x_test.reshape(10000, 3072)
		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')
		x_train /= 255
		x_test /= 255
		y_train = to_categorical(y_train, nbClasses)
		y_test = to_categorical(y_test, nbClasses)

		return x_train, y_train, x_test, y_test




	'''
    Train the dataset
    Returns the accuracy
    '''
	def getFitness(self, net, genNum, dataset):
		self.fitness = {}
		self.network = {}
		self.network = net

		i = 0;		j = 0;

		# set fitness for all the networks = 0
		for i in range(0, self.numNetworks):
			self.fitness[i] = 0


		# evaluate fitness of all the networks
		for i in range(self.numNetworks):

			batchSize = 64
			input_shape = (3072,)

			# Get number of classification classes
			nbClasses = self.getnbClasses(dataset)

			# Fetch details of the dataset
			x_train, y_train, x_test, y_test = self.getData(dataset, nbClasses)


			# Get details of the neural network to be designed
			activation = self.network[i]['activation']
			optimizer = self.network[i]['optimizer']
			nbNeurons = self.network[i]['nbNeurons']
			nbLayers = self.network[i]['nbLayers']
			dropout = self.network[i]['dropout']

			# Initializes the model type to be trained
			model = Sequential()

			# Create the neural network
			# Add the layers in the neural network
			for j in range(nbLayers):
				if j == 0:
					model.add(Dense(nbNeurons, activation=activation, input_shape=input_shape))
				else:
					model.add(Dense(nbNeurons, activation=activation))
				model.add(Dropout(dropout))
			model.add(Dense(nbClasses, activation='softmax'))

			# Compile the model
			model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

			# Fit the model
			model.fit(x_train, y_train, batch_size=batchSize, epochs=1, verbose=0, validation_data=(x_test, y_test),
					  callbacks=[earlyStopper])

			# Get the fitness of the model
			# Accuracy and Error
			y = model.evaluate(x_test, y_test, verbose=0)

			self.fitness[i] = y[1] * 100


		# Return the accuracy

		return self.fitness




class compare():
	'''
    Initializes the fitness comparison class
    '''

	def __init__(self, networkFitness):
		self.networkFitness = networkFitness


	def networkData(self, numNetworks, fitnessParent, fitnessChild, QnetC, QnetP):

		for n in range(numNetworks):
			if (self.networkFitness[n] < fitnessParent[n]) or (self.networkFitness[n] < fitnessChild[n]):
				if fitnessParent[n] > fitnessChild[n]:
					self.networkFitness[n] = fitnessParent[n]
				else:
					self.networkFitness[n] = fitnessChild[n]
					QnetP[n] = QnetC[n]

		return self.networkFitness, QnetP



	def BestAndPoorest(self):

		maxIndex = max(self.networkFitness, key=self.networkFitness.get)
		minIndex = min(self.networkFitness, key=self.networkFitness.get)

		return maxIndex, minIndex


	'''
	To get the best fitness of the generation	
	Returns the best fitness of the generation and the data
	'''
	def genFitness(self, numNetworks, numParam, param, QnetParent):

		# Get the generation best fitness, if it's better then proceed
		maxIndex, minIndex = self.BestAndPoorest()


		# The network with the worst fitness is deleted and reinitialized looking for better possibilities
		# The vector and the measured values for the network with minimum fitness are:
		cv, QnetNew = qgaNetwork(numNetworks, numParam, param).Init_population()

		QnetParent[minIndex] = QnetNew[minIndex]

		return self.networkFitness[maxIndex], QnetParent

