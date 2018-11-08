import math
import numpy as np

import random

from keras.datasets import cifar10, cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

import qNetwork

'''
Contains functions of Genetic ALgorithm :
	Natural Selection
	Crossover
	Mutation
	Breeding = Natural Selection + Crossover + Mutation
	Accuracy Measurement
'''


class qgaGenetic(qNetwork.qgaNetwork):


	'''
	Initialize the class inheriting the properties of the "network" class
	'''
	def __init__(self, numNetworks, numParam, param=None):

		qNetwork.qgaNetwork.__init__(self, numNetworks, numParam, param=param)



	'''
	Natural Selection
	get parents details
		Mum is the current network
		Dad is the previous Network
		Returns parents
	'''
	def getParents(self, i, numNetworks):
		mum = i
		dad = ((i + self.numNetworks - 1) % self.numNetworks)
		return mum, dad




	'''
	Crossover:
		Multiple point crossover
		Randomly choose the value of every position
		And substitute the value with the value of any of the parent
		Returns the network of the child after crossover
	'''
	def crossover(self, i, mum, dad, net):

		for p in range(self.numParam):
			rand = random.randint(0, 1)

			if rand == 0:
				self.nqpv[i, p, 0] = net[mum,p,0]
				self.nqpv[i, p, 1] = net[mum,p,1]
			else:
				self.nqpv[i, p, 0] = net[dad,p,0]
				self.nqpv[i, p, 1] = net[dad,p,1]

			rand = None
		return self.nqpv



	'''
	Mutation:
		Keeping mutation chance to be 30%
		Mutate one of the params
		Returns the network of the child after mutation
	'''
	def mutation(self, i, qcv, mutationChance):

		# probability of mutation
		popProbab=(np.random.random_integers(100))/100

		if popProbab<=mutationChance:

			# the value of which is to be mutated
			# optimizer, e.g.
			netProbab=np.random.randint(0, 4)

			for j in range(0,netProbab):
				self.nqpv[i,j,0]=qcv[i,j,1]
				self.nqpv[i,j,1]=qcv[i,j,0]

		# npqv has the child after breeding
		# return the child
		return self.nqpv



	'''
	Breeding
	Natural Selection, Crossover & Mutation
	Returns the network of the child
	'''
	def breeding(self, net, mutationChance):

		for i in range(self.numNetworks):

			# NATURAL SELECTION
			mum, dad = self.getParents(i, self.numNetworks)

			# CROSSOVER
			child = self.crossover(i, mum, dad, net)

			# MUTATION
			child = self.mutation(i, child, mutationChance)

		return child



