import math
import numpy as np

import collections
from collections import OrderedDict

import logging
import random


import qGenetic as qg
import qNetwork as qn


'''
Get parameters for optimizing Neural Network
'''
def getParameters():
	# Number of generations
	generation = 25
	# Dataset for comparison
	dataset = 'cifar10'
	# Number of networks OR population size in every generations
	numNetworks = 12
	# Rate of mutation
	mutationChance = 30
	# Hyper-parameters to be optimized
	param = collections.OrderedDict({
		'nbNeurons': {1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 128},
		'nbLayers': {1: 1, 2: 3, 3: 6, 4: 9, 5: 12, 6: 15},
		'activation': {1: 'sigmoid', 2: 'elu', 3: 'selu', 4: 'relu', 5: 'tanh', 6: 'hard_sigmoid'},
		'optimizer': {1: 'sgd', 2: 'nadam', 3: 'adagrad', 4: 'adadelta', 5: 'adam', 6: 'adamax'},
		'dropout': {1: 0.1, 2: 0.2, 3: 0.25, 4: 0.3, 5: 0.4, 6: 0.5}
	})
	# Number of hyper-parameters
	numParam = len(param)

	return generation, dataset, numNetworks, mutationChance, param, numParam


'''
Initialize the classes
'''
def initClasses(param, numNetworks, numParam, networkFitness):
	# The Network class
	qgaNet = qn.qgaNetwork(numNetworks, numParam, param)
	# The Genetic Algorithm class
	qgaGen = qg.qgaGenetic(numNetworks, numParam, param)
	# The class to calculate the fitness of the networks
	qgaFit = qn.fitness(numNetworks)
	# The class with comparison functions
	qgaCom = qn.compare(networkFitness)
	return qgaNet, qgaGen, qgaFit, qgaCom


'''
Neural Network Tuning Inspired by Quantum Algorithm
'''
#@profile
def QNNT():

	# Get Parameters
	generation, dataset, numNetworks, mutationChance, param, numParam = getParameters()


	# Get the logger
	filename = 'output.log'
	logger = logging.getLogger()
	handler = logging.FileHandler(filename)
	handler.setLevel(logging.DEBUG)
	formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	logger.setLevel(logging.DEBUG)

	# Initialize the population & the fitness
	qpv = {};				# The quantum Parent network - Before Measurement
	qcv = {};				# The quantum Child network - Before Measurement
	QnetParent = {};		# The quantum networks - Parents - Measured Value
	QnetChild = {};			# The quantum networks - Children - Measured Value
	fitnessParent = {};		# The fitness array of the parent
	fitnessChild = {}; 		# The fitness array of the child
	networkFitness = {};	# Array of the better out of two fitness - parent and child
	for n in range(numNetworks):
		networkFitness[n] = -1
	genBestFitness = -1; 	# Best of all the network fitness over the generations


	# Initialize the classes
	qgaNet, qgaGen, qgaFit, qgaCom = initClasses(param, numNetworks, numParam, networkFitness)


	# Initialize the population
	# Randomly initialize the quantum network array and it's measured value
	qpv, QnetParent = qgaNet.Init_population()


	for g in range(generation):

		if genBestFitness < 100:


			# GET PARENT FITNESS/ACCURACY
			fitnessParent = qgaFit.getFitness(QnetParent, g, dataset)

			# BREED THE CHILD
			qcv = qgaGen.breeding(qpv, mutationChance)

			# MAKE THE MEASUREMENT OF THE CHILD
			QnetChild = qgaNet.Measure(qcv)

			# GET CHILD'S FITNESS/ACCURACY
			try:
				fitnessChild = qgaFit.getFitness(QnetChild, g, dataset)
			except KeyError:
				print(qpv, qcv)

			'''
			If the network fitness has improved over previous generation, 
				then pass on the features/hyperparameters
			Pass on the better of the two (parent or child) from this generation to the next generation
			'''
			networkFitness, data = qgaCom.networkData(numNetworks, fitnessParent, fitnessChild, QnetChild, QnetParent)


			'''
			Compare the fitness of the best networks of all the families
			Get the best fitness the generation 
			Kill the poorest performing of the population 
			Randomly initialize the poorest fitness population to keep the population constant
			'''
			genBestFitness, QnetParent = qgaCom.genFitness(numNetworks, numParam, param, QnetParent)

			logger.debug('generation=%d, parent=%s, child=%s, parentFitness=%s, childFitness=%s, networkFitness=%s, genBestFitness=%0.4f',
						 g, QnetParent, QnetChild, fitnessParent, fitnessChild, networkFitness, genBestFitness)

		else:
			print('Best Fitness reached:', genBestFitness)

if __name__ == '__main__':
	QNNT()


