import network
import genetic
import parallel

import socket

import logging

import collections
from collections import OrderedDict

import mpi4py
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
mpi4py.rc(initialize=False, finalize=False)



# import unittest
# from randomdict import RandomDict
# import warnings
# warnings.filterwarnings("always")
# from memory_profiler import profile


'''
Initialize the classes
'''
def initClasses(param, MPI, groupSize, networkFitness):
	# The Network class
	net = network.Network(param)
	# The Genetic Algorithm class
	ga = genetic.geneticAlgorithm(param)
	# The class with comparison functions
	com = network.compare(networkFitness)
	# The MPI class
	pd = parallel.parallelDistributed(MPI, groupSize,param) 
	return net, ga, com, pd



'''
Get parameters for optimizing Neural Network
'''
def getParameters():
	# Number of generations
	generation = 25
	# Dataset for comparison
	dataset = 'cifar10'
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
	# The size of the Island i.e. population size per island
	groupSize = 4
	return generation, dataset, mutationChance, param, groupSize



'''
Distributed Neural Network Tuning
'''
#@profile
def INNT():

	# Initializing the MPI and testing if it's been initialized
	MPI.Init()
	print(MPI.Is_initialized())
	print(MPI.Is_finalized())


	# Get Parameters
	generation, dataset, mutationChance, param, groupSize = getParameters()


	# Initialize the fitness
	fitnessParent = -1; 	# The fitness of the parent
	fitnessChild = -1;		# The fitness of the child
	networkFitness = -1;	# The fitness of the network
	genBestFitness = -1;	# Fitness of the generation



	# Initialize the classes
	net, ga, com, pd = initClasses(param, MPI, groupSize, networkFitness)


	# Get the logger
	# filename = 'output{}.log'.format(pd.rank)
	filename = 'output.log'
	logger = logging.getLogger()
	handler = logging.FileHandler(filename)
	handler.setLevel(logging.DEBUG)
	formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	logger.setLevel(logging.DEBUG)



	# Split the communicator
	subGroup = pd.rank / groupSize
	subComm = MPI.Comm.Split(MPI.COMM_WORLD, subGroup, pd.rank)


	# initialize the networks
	# one random network at every processor

	# INITIALIZZEEEE ISLAND WITH SOME SPECIALITYYYYY
	data = net.initNetwork()

	# Islands differ in activation function
	# Since there will be at min 2 subgroups
	if pd.subGroup == 0:
		data['activation'] = 'sigmoid'
	elif pd.subGroup == 1:
		data['activation'] = 'elu'
	else:
		data['activation'] = 'selu'
		

	# Start running GA (Genetic Algorithm) generation
	for g in range(generation):

		if genBestFitness < 100:

			# GET PARENT FITNESS/ACCURACY
			# Every processor trains and evaluate the accuracy/fitness of the parent network
			fitnessParent = ga.getFitness(data, dataset)
			print('loop_1 done', g, pd.rank)

			# BREED THE CHILD
			# This to be done using MPI ISend
			# Get the parent using Non Blocking exchange
			child = ga.breeding(pd.rank, g, data, mutationChance, pd.intraIslandExchange(data, subComm))
			MPI.COMM_WORLD.Barrier()
		

			# GET CHILD'S FITNESS/ACCURACY
			# Every processor trains and evaluate the accuracy/fitness of the child network
			fitnessChild = ga.getFitness(child, dataset)

			'''
			If the network fitness has improved over previous generation, 
				then pass on the features/hyperparameters
			Pass on the better of the two (parent or child) from this generation to the next generation
			Comparison done - of the previous value at the procecssor with the new computed value
			'''
			networkFitness, data = com.networkData(fitnessParent, fitnessChild, data, child)


			'''
			Compare the fitness of the best networks of all the families
			Compares the fitness of all the networks data that are with all the processors in the communication
			Get the best fitness the generation 
			Kill the poorest performing of the population 
			Randomly initialize the poorest fitness population to keep the population constant
			'''
			genBestFitness, data = com.genFitness(data, param, MPI, groupSize)
			# print(genBestFitness, data)

			logger.debug('generation=%d, Rank=%d, processid=%s, group=ID%d, subRank=%d, parent=%s, child=%s, parentFitness=%0.4f, childFitness=%0.4f, networkFitness=%0.4f, genBestFitness=%0.4f', g, pd.rank, socket.gethostname(), pd.subGroup, subComm.Get_rank(), data, child, fitnessParent, fitnessChild, networkFitness, genBestFitness)


			'''
			Do inter-island exchange after every 5 generations
				In this all the ranks are sending the data to the previous ranks
			'''
			if g%5==0:
				pd.interIslandExchange(data, subComm)
			print('loop_6 done', pd.rank)
			MPI.COMM_WORLD.Barrier()

		else:
			# Broadcast the best results to all the processors
			pd.broadcast(data, pd.rank)
			print('best fitness achieved')
			# And halt
			MPI.Finalize()

	MPI.Finalize()



if __name__ == '__main__':
	INNT()
		



