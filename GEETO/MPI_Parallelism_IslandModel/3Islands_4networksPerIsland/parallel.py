import mpi4py
mpi4py.rc(initialize=False, finalize=False)


'''
Class for parallel functions:
	Initializes the MPI features
		Non-Blocking Exchange
		Broadcast
'''
class parallelDistributed():

	def __init__(self, MPI, groupSize, param=None):
		self.mpi = MPI
		self.comm=MPI.COMM_WORLD
		self.size = self.comm.Get_size()
		# assert comm.size > 1
		self.rank = self.comm.Get_rank()
		self.name = MPI.Get_processor_name()

		'''
		For Island model
			3 islands of population size 4 each have been created and tested
			splitting the communicator in groups of 3
		'''
		self.groupSize = groupSize
		self.group = self.comm.Get_group()
		self.subGroup = int(self.rank / groupSize)
		#self.subComm = MPI.Comm.Split(self.comm, self.subGroup, self.rank)
		#self.subSize, self.subRank = self.subComm.Get_size(), self.subComm.Get_rank()




	'''
	Non-Blocking Exchange
		Send the params to the next network
		Receive params of the previous network
		(size + rank - 1)%size //previous // recv from prev
		(size + rank + 1)%size //next // send to next
		Returns the data of the previous network
	'''
	def nonBlockingExchange(self,data):
		reqSend1 = self.comm.isend(data, dest=((self.size+self.rank+1)%self.size), tag=self.rank)
		reqRecv2 = self.comm.irecv(source=((self.size+self.rank-1)%self.size), tag=self.rank-1)
		dataPrev = reqRecv2.wait()
		reqSend1.wait()
		return dataPrev


	'''
	Get the details of the network with the minimum fitness
	the output has two values - 
		the min fitness value
		the rank of the processor having min fitness
	'''
	def getMin(self, msg):
		minloc = self.comm.allreduce(sendobj=(msg, self.rank), op=self.mpi.MINLOC)
		# minloc[0] will have the data
		# minloc[1] will have the rank
		print minloc[1], 'checking', minloc[0]
		return minloc



	'''
	Get the details of the network with the minimum fitness
	the output has two values - 
		the min fitness value
		the rank of the processor having min fitness
	'''
	def getMax(self, msg):
		maxloc = self.comm.allreduce(sendobj=(msg, self.rank), op=self.mpi.MAXLOC)
		# maxloc[0] will have the data
		# maxloc[1] will have the rank
		print maxloc[1], 'checking', maxloc[0]
		return maxloc


	'''
	Get the rank
	'''
	def getCommRank(self):
		return self.comm.Get_rank()


	'''
	Broadcast:
		Data Broadcasted from the root to all the other sockets/nodes
	'''
	def broadcast(self, data, theRank):
		data = self.comm.bcast(data, root = theRanm)
		return data



	'''
	 Intra Island Exchange
	Exhchange within the island
		Send the params to the next network within the island
		Receive params of the previous network within the island
		Returns the data of the previous network
	Send to prev rank making sure, the subGroup of the previous rank is same as the subGroup of the present rank

	'''
	def intraIslandExchange(self,data, subComm):
		subSize, subRank = subComm.Get_size(), subComm.Get_rank()

		reqSend1 = subComm.isend(data, dest=((subSize+subRank+1)%subSize), tag=subRank)
		reqRecv2 = subComm.irecv(source=((subSize+subRank-1)%subSize), tag=subRank-1)

		dataPrev = reqRecv2.wait()
		reqSend1.wait()

		return dataPrev



	'''
	Inter Island Exchange
	Exchange between islands
	the last rank holder is sent to the other island
		within every island/subGroup, network having the zeroeth subRank is sent to the next island's last subRank
		within every island/subGroup, processor with the zeroeth subRank receives from the previous island's last subRank
	'''
	def interIslandExchange(self, data, subComm):
		subSize, subRank = subComm.Get_size(), subComm.Get_rank()
		
		if subRank == 0:
			sendTo = self.sendToIsland(subSize)
			recvFrom = self.recvFromIsland(subSize)
			print(self.rank, sendTo, recvFrom)

			reqSend1 = self.comm.isend(data, dest=sendTo, tag=recvFrom)
			reqRecv2 = self.comm.irecv(source=recvFrom, tag=sendTo)

			dataPrev = reqRecv2.wait()
			reqSend1.wait()
			return dataPrev


	'''
	Get all the ranks of a best fitness networks within each of the subgroup
	'''
	def getBestRanks(self):
		lst = []
		start = self.subGroup
		lst.extend(str(start))
		for _ in range(self.subGroup):
			lst.extend(str(start+3))  
		return lst


	'''
	Get te rank from where to receive
	'''
	def recvFromIsland(self, subSize):
		recvFrom = self.rank - self.groupSize
		if recvFrom < 0: #self.groupSize:
			recvFrom = (int((self.size-1)/subSize))*subSize
		return recvFrom

	'''
	Get the rank where the data is to be send
	'''
	def sendToIsland(self, subSize):
		toRank = self.rank + subSize
		if toRank > self.size-1:
			toRank = 0
		return toRank
